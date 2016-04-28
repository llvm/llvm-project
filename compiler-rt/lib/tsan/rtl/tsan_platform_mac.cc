//===-- tsan_platform_mac.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Mac-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_MAC

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_flags.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

namespace __tsan {

#ifndef SANITIZER_GO
static void *SignalSafeGetOrAllocate(uptr *dst, uptr size) {
  atomic_uintptr_t *a = (atomic_uintptr_t *)dst;
  void *val = (void *)atomic_load_relaxed(a);
  atomic_signal_fence(memory_order_acquire);  // Turns the previous load into
                                              // acquire wrt signals.
  if (UNLIKELY(val == nullptr)) {
    val = (void *)internal_mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(val);
    void *cmp = nullptr;
    if (!atomic_compare_exchange_strong(a, (uintptr_t *)&cmp, (uintptr_t)val,
                                        memory_order_acq_rel)) {
      internal_munmap(val, size);
      val = cmp;
    }
  }
  return val;
}

// On OS X, accessing TLVs via __thread or manually by using pthread_key_* is
// problematic, because there are several places where interceptors are called
// when TLVs are not accessible (early process startup, thread cleanup, ...).
// The following provides a "poor man's TLV" implementation, where we use the
// shadow memory of the pointer returned by pthread_self() to store a pointer to
// the ThreadState object. The main thread's ThreadState pointer is stored
// separately in a static variable, because we need to access it even before the
// shadow memory is set up.
static uptr main_thread_identity = 0;
static ThreadState *main_thread_state = nullptr;

ThreadState *cur_thread() {
  ThreadState **fake_tls;
  uptr thread_identity = (uptr)pthread_self();
  if (thread_identity == main_thread_identity || main_thread_identity == 0) {
    fake_tls = &main_thread_state;
  } else {
    fake_tls = (ThreadState **)MemToShadow(thread_identity);
  }
  ThreadState *thr = (ThreadState *)SignalSafeGetOrAllocate(
      (uptr *)fake_tls, sizeof(ThreadState));
  return thr;
}

// TODO(kuba.brecka): This is not async-signal-safe. In particular, we call
// munmap first and then clear `fake_tls`; if we receive a signal in between,
// handler will try to access the unmapped ThreadState.
void cur_thread_finalize() {
  uptr thread_identity = (uptr)pthread_self();
  if (thread_identity == main_thread_identity) {
    // Calling dispatch_main() or xpc_main() actually invokes pthread_exit to
    // exit the main thread. Let's keep the main thread's ThreadState.
    return;
  }
  ThreadState **fake_tls = (ThreadState **)MemToShadow(thread_identity);
  internal_munmap(*fake_tls, sizeof(ThreadState));
  *fake_tls = nullptr;
}
#endif

uptr GetShadowMemoryConsumption() {
  return 0;
}

void FlushShadowMemory() {
}

void WriteMemoryProfile(char *buf, uptr buf_size, uptr nthread, uptr nlive) {
}

#ifndef SANITIZER_GO
void InitializeShadowMemoryPlatform() { }

// On OS X, GCD worker threads are created without a call to pthread_create. We
// need to properly register these threads with ThreadCreate and ThreadStart.
// These threads don't have a parent thread, as they are created "spuriously".
// We're using a libpthread API that notifies us about a newly created thread.
// The `thread == pthread_self()` check indicates this is actually a worker
// thread. If it's just a regular thread, this hook is called on the parent
// thread.
typedef void (*pthread_introspection_hook_t)(unsigned int event,
                                             pthread_t thread, void *addr,
                                             size_t size);
extern "C" pthread_introspection_hook_t pthread_introspection_hook_install(
    pthread_introspection_hook_t hook);
static const uptr PTHREAD_INTROSPECTION_THREAD_CREATE = 1;
static const uptr PTHREAD_INTROSPECTION_THREAD_TERMINATE = 3;
static pthread_introspection_hook_t prev_pthread_introspection_hook;
static void my_pthread_introspection_hook(unsigned int event, pthread_t thread,
                                          void *addr, size_t size) {
  if (event == PTHREAD_INTROSPECTION_THREAD_CREATE) {
    if (thread == pthread_self()) {
      // The current thread is a newly created GCD worker thread.
      ThreadState *thr = cur_thread();
      Processor *proc = ProcCreate();
      ProcWire(proc, thr);
      ThreadState *parent_thread_state = nullptr;  // No parent.
      int tid = ThreadCreate(parent_thread_state, 0, (uptr)thread, true);
      CHECK_NE(tid, 0);
      ThreadStart(thr, tid, GetTid());
    }
  } else if (event == PTHREAD_INTROSPECTION_THREAD_TERMINATE) {
    if (thread == pthread_self()) {
      ThreadState *thr = cur_thread();
      if (thr->tctx) {
        DestroyThreadState();
      }
    }
  }

  if (prev_pthread_introspection_hook != nullptr)
    prev_pthread_introspection_hook(event, thread, addr, size);
}
#endif

void InitializePlatformEarly() {
}

void InitializePlatform() {
  DisableCoreDumperIfNecessary();
#ifndef SANITIZER_GO
  CheckAndProtect();

  CHECK_EQ(main_thread_identity, 0);
  main_thread_identity = (uptr)pthread_self();

  prev_pthread_introspection_hook =
      pthread_introspection_hook_install(&my_pthread_introspection_hook);
#endif
}

#ifndef SANITIZER_GO
// Note: this function runs with async signals enabled,
// so it must not touch any tsan state.
int call_pthread_cancel_with_cleanup(int(*fn)(void *c, void *m,
    void *abstime), void *c, void *m, void *abstime,
    void(*cleanup)(void *arg), void *arg) {
  // pthread_cleanup_push/pop are hardcore macros mess.
  // We can't intercept nor call them w/o including pthread.h.
  int res;
  pthread_cleanup_push(cleanup, arg);
  res = fn(c, m, abstime);
  pthread_cleanup_pop(0);
  return res;
}
#endif

bool IsGlobalVar(uptr addr) {
  return false;
}

}  // namespace __tsan

#endif  // SANITIZER_MAC
