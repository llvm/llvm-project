//===-- scudo_tls_android.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo thread local structure implementation for Android.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_LINUX && SANITIZER_ANDROID

#include "scudo_tls.h"

#include <pthread.h>

namespace __scudo {

static pthread_once_t GlobalInitialized = PTHREAD_ONCE_INIT;
static pthread_key_t PThreadKey;

static atomic_uint32_t ThreadContextCurrentIndex;
static ScudoThreadContext *ThreadContexts;
static uptr NumberOfContexts;

// sysconf(_SC_NPROCESSORS_{CONF,ONLN}) cannot be used as they allocate memory.
static uptr getNumberOfCPUs() {
  cpu_set_t CPUs;
  CHECK_EQ(sched_getaffinity(0, sizeof(cpu_set_t), &CPUs), 0);
  return CPU_COUNT(&CPUs);
}

static void initOnce() {
  // Hack: TLS_SLOT_TSAN was introduced in N. To be able to use it on M for
  // testing, we create an unused key. Since the key_data array follows the tls
  // array, it basically gives us the extra entry we need.
  // TODO(kostyak): remove and restrict to N and above.
  CHECK_EQ(pthread_key_create(&PThreadKey, NULL), 0);
  initScudo();
  NumberOfContexts = getNumberOfCPUs();
  ThreadContexts = reinterpret_cast<ScudoThreadContext *>(
      MmapOrDie(sizeof(ScudoThreadContext) * NumberOfContexts, __func__));
  for (uptr i = 0; i < NumberOfContexts; i++)
    ThreadContexts[i].init();
}

void initThread() {
  pthread_once(&GlobalInitialized, initOnce);
  // Initial context assignment is done in a plain round-robin fashion.
  u32 Index = atomic_fetch_add(&ThreadContextCurrentIndex, 1,
                               memory_order_relaxed);
  ScudoThreadContext *ThreadContext =
      &ThreadContexts[Index % NumberOfContexts];
  *get_android_tls_ptr() = reinterpret_cast<uptr>(ThreadContext);
}

ScudoThreadContext *getThreadContextAndLockSlow() {
  ScudoThreadContext *ThreadContext;
  // Go through all the contexts and find the first unlocked one. 
  for (u32 i = 0; i < NumberOfContexts; i++) {
    ThreadContext = &ThreadContexts[i];
    if (ThreadContext->tryLock()) {
      *get_android_tls_ptr() = reinterpret_cast<uptr>(ThreadContext);
      return ThreadContext;
    }
  }
  // No luck, find the one with the lowest precedence, and slow lock it.
  u64 Precedence = UINT64_MAX;
  for (u32 i = 0; i < NumberOfContexts; i++) {
    u64 SlowLockPrecedence = ThreadContexts[i].getSlowLockPrecedence();
    if (SlowLockPrecedence && SlowLockPrecedence < Precedence) {
      ThreadContext = &ThreadContexts[i];
      Precedence = SlowLockPrecedence;
    }
  }
  if (LIKELY(Precedence != UINT64_MAX)) {
    ThreadContext->lock();
    *get_android_tls_ptr() = reinterpret_cast<uptr>(ThreadContext);
    return ThreadContext;
  }
  // Last resort (can this happen?), stick with the current one.
  ThreadContext =
      reinterpret_cast<ScudoThreadContext *>(*get_android_tls_ptr());
  ThreadContext->lock();
  return ThreadContext;
}

}  // namespace __scudo

#endif  // SANITIZER_LINUX && SANITIZER_ANDROID
