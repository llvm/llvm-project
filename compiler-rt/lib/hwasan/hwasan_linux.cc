//===-- hwasan_linux.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Linux-, NetBSD- and FreeBSD-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD

#include "hwasan.h"
#include "hwasan_thread.h"

#include <elf.h>
#include <link.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <unwind.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_procmaps.h"

uptr __hwasan_shadow_memory_dynamic_address;

__attribute__((alias("__hwasan_shadow_memory_dynamic_address")))
extern uptr __hwasan_shadow_memory_dynamic_address_internal;

namespace __hwasan {

bool InitShadow() {
  const uptr maxVirtualAddress = GetMaxUserVirtualAddress();
  uptr shadow_size = MEM_TO_SHADOW_OFFSET(maxVirtualAddress) + 1;
  __hwasan_shadow_memory_dynamic_address =
      reinterpret_cast<uptr>(MmapNoReserveOrDie(shadow_size, "shadow"));
  return true;
}

static void HwasanAtExit(void) {
  if (flags()->print_stats && (flags()->atexit || hwasan_report_count > 0))
    ReportStats();
  if (hwasan_report_count > 0) {
    // ReportAtExitStatistics();
    if (common_flags()->exitcode)
      internal__exit(common_flags()->exitcode);
  }
}

void InstallAtExitHandler() {
  atexit(HwasanAtExit);
}

// ---------------------- TSD ---------------- {{{1

static pthread_key_t tsd_key;
static bool tsd_key_inited = false;

void HwasanTSDInit(void (*destructor)(void *tsd)) {
  CHECK(!tsd_key_inited);
  tsd_key_inited = true;
  CHECK_EQ(0, pthread_key_create(&tsd_key, destructor));
}

HwasanThread *GetCurrentThread() {
  return (HwasanThread*)pthread_getspecific(tsd_key);
}

void SetCurrentThread(HwasanThread *t) {
  // Make sure that HwasanTSDDtor gets called at the end.
  CHECK(tsd_key_inited);
  // Make sure we do not reset the current HwasanThread.
  CHECK_EQ(0, pthread_getspecific(tsd_key));
  pthread_setspecific(tsd_key, (void *)t);
}

void HwasanTSDDtor(void *tsd) {
  HwasanThread *t = (HwasanThread*)tsd;
  if (t->destructor_iterations_ > 1) {
    t->destructor_iterations_--;
    CHECK_EQ(0, pthread_setspecific(tsd_key, tsd));
    return;
  }
  // Make sure that signal handler can not see a stale current thread pointer.
  atomic_signal_fence(memory_order_seq_cst);
  HwasanThread::TSDDtor(tsd);
}

struct AccessInfo {
  uptr addr;
  uptr size;
  bool is_store;
  bool is_load;
};

#if defined(__aarch64__)
static AccessInfo GetAccessInfo(siginfo_t *info, ucontext_t *uc) {
  AccessInfo ai;
  uptr pc = (uptr)info->si_addr;

  struct {
    uptr addr;
    unsigned size;
    bool is_store;
  } handlers[] = {
      {(uptr)&__hwasan_load1, 1, false},   {(uptr)&__hwasan_load2, 2, false},
      {(uptr)&__hwasan_load4, 4, false},   {(uptr)&__hwasan_load8, 8, false},
      {(uptr)&__hwasan_load16, 16, false},  {(uptr)&__hwasan_load, 0, false},
      {(uptr)&__hwasan_store1, 1, true},  {(uptr)&__hwasan_store2, 2, true},
      {(uptr)&__hwasan_store4, 4, true},  {(uptr)&__hwasan_store8, 8, true},
      {(uptr)&__hwasan_store16, 16, true}, {(uptr)&__hwasan_store, 0, true}};
  int best = -1;
  uptr best_distance = 0;
  for (size_t i = 0; i < sizeof(handlers) / sizeof(handlers[0]); ++i) {
    uptr handler = handlers[i].addr;
    // Don't accept pc == handler: HLT is never the first instruction.
    if (pc <= handler) continue;
    uptr distance = pc - handler;
    if (distance > 256) continue;
    if (best == -1 || best_distance > distance) {
      best = i;
      best_distance = distance;
    }
  }

  // Not ours.
  if (best == -1)
    return AccessInfo{0, 0, false, false};

  ai.is_store = handlers[best].is_store;
  ai.is_load = !handlers[best].is_store;
  ai.size = handlers[best].size;

  ai.addr = uc->uc_mcontext.regs[0];
  if (ai.size == 0)
    ai.size = uc->uc_mcontext.regs[1];
  return ai;
}
#else
static AccessInfo GetAccessInfo(siginfo_t *info, ucontext_t *uc) {
  return AccessInfo{0, 0, false, false};
}
#endif

static void HwasanOnSIGILL(int signo, siginfo_t *info, ucontext_t *uc) {
  SignalContext sig{info, uc};
  AccessInfo ai = GetAccessInfo(info, uc);
  if (!ai.is_store && !ai.is_load)
    return;

  InternalScopedBuffer<BufferedStackTrace> stack_buffer(1);
  BufferedStackTrace *stack = stack_buffer.data();
  stack->Reset();
  GetStackTrace(stack, kStackTraceMax, sig.pc, sig.bp, uc,
                common_flags()->fast_unwind_on_fatal);

  ReportTagMismatch(stack, ai.addr, ai.size, ai.is_store);

  ++hwasan_report_count;
  if (flags()->halt_on_error)
    Die();
  else
    uc->uc_mcontext.pc += 4;
}

static void OnStackUnwind(const SignalContext &sig, const void *,
                          BufferedStackTrace *stack) {
  GetStackTrace(stack, kStackTraceMax, sig.pc, sig.bp, sig.context,
                common_flags()->fast_unwind_on_fatal);
}

void HwasanOnDeadlySignal(int signo, void *info, void *context) {
  // Probably a tag mismatch.
  // FIXME: detect pc range in __hwasan_load* or __hwasan_store*.
  if (signo == SIGILL)
    HwasanOnSIGILL(signo, (siginfo_t *)info, (ucontext_t*)context);
  else
    HandleDeadlySignal(info, context, GetTid(), &OnStackUnwind, nullptr);
}


} // namespace __hwasan

#endif // SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD
