//===-- hwasan_linux.cc -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file is a part of HWAddressSanitizer and contains Linux-, NetBSD- and
/// FreeBSD-specific code.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD

#include "hwasan.h"
#include "hwasan_dynamic_shadow.h"
#include "hwasan_interface_internal.h"
#include "hwasan_mapping.h"
#include "hwasan_thread.h"

#include <elf.h>
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <unwind.h>

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_procmaps.h"

namespace __hwasan {

static void ReserveShadowMemoryRange(uptr beg, uptr end, const char *name) {
  CHECK_EQ((beg % GetMmapGranularity()), 0);
  CHECK_EQ(((end + 1) % GetMmapGranularity()), 0);
  uptr size = end - beg + 1;
  DecreaseTotalMmap(size);  // Don't count the shadow against mmap_limit_mb.
  void *res = MmapFixedNoReserve(beg, size, name);
  if (res != (void *)beg) {
    Report(
        "ReserveShadowMemoryRange failed while trying to map 0x%zx bytes. "
        "Perhaps you're using ulimit -v\n",
        size);
    Abort();
  }
  if (common_flags()->no_huge_pages_for_shadow) NoHugePagesInRegion(beg, size);
  if (common_flags()->use_madv_dontdump) DontDumpShadowMemory(beg, size);
}

static void ProtectGap(uptr addr, uptr size) {
  if (!size)
    return;
  void *res = MmapFixedNoAccess(addr, size, "shadow gap");
  if (addr == (uptr)res)
    return;
  // A few pages at the start of the address space can not be protected.
  // But we really want to protect as much as possible, to prevent this memory
  // being returned as a result of a non-FIXED mmap().
  if (addr == 0) {
    uptr step = GetMmapGranularity();
    while (size > step) {
      addr += step;
      size -= step;
      void *res = MmapFixedNoAccess(addr, size, "shadow gap");
      if (addr == (uptr)res)
        return;
    }
  }

  Report(
      "ERROR: Failed to protect shadow gap [%p, %p]. "
      "HWASan cannot proceed correctly. ABORTING.\n", (void *)addr,
      (void *)(addr + size));
  DumpProcessMap();
  Die();
}

static uptr kLowMemStart;
static uptr kLowMemEnd;
static uptr kLowShadowEnd;
static uptr kLowShadowStart;
static uptr kHighShadowStart;
static uptr kHighShadowEnd;
static uptr kHighMemStart;
static uptr kHighMemEnd;

static void PrintRange(uptr start, uptr end, const char *name) {
  Printf("|| [%p, %p] || %.*s ||\n", (void *)start, (void *)end, 10, name);
}

static void PrintAddressSpaceLayout() {
  PrintRange(kHighMemStart, kHighMemEnd, "HighMem");
  if (kHighShadowEnd + 1 < kHighMemStart)
    PrintRange(kHighShadowEnd + 1, kHighMemStart - 1, "ShadowGap");
  else
    CHECK_EQ(kHighShadowEnd + 1, kHighMemStart);
  PrintRange(kHighShadowStart, kHighShadowEnd, "HighShadow");
  if (SHADOW_OFFSET) {
    if (kLowShadowEnd + 1 < kHighShadowStart)
      PrintRange(kLowShadowEnd + 1, kHighShadowStart - 1, "ShadowGap");
    else
      CHECK_EQ(kLowMemEnd + 1, kHighShadowStart);
    PrintRange(kLowShadowStart, kLowShadowEnd, "LowShadow");
    if (kLowMemEnd + 1 < kLowShadowStart)
      PrintRange(kLowMemEnd + 1, kLowShadowStart - 1, "ShadowGap");
    else
      CHECK_EQ(kLowMemEnd + 1, kLowShadowStart);
    PrintRange(kLowMemStart, kLowMemEnd, "LowMem");
    CHECK_EQ(0, kLowMemStart);
  } else {
    if (kLowMemEnd + 1 < kHighShadowStart)
      PrintRange(kLowMemEnd + 1, kHighShadowStart - 1, "ShadowGap");
    else
      CHECK_EQ(kLowMemEnd + 1, kHighShadowStart);
    PrintRange(kLowMemStart, kLowMemEnd, "LowMem");
    CHECK_EQ(kLowShadowEnd + 1, kLowMemStart);
    PrintRange(kLowShadowStart, kLowShadowEnd, "LowShadow");
    PrintRange(0, kLowShadowStart - 1, "ShadowGap");
  }
}

static uptr GetHighMemEnd() {
  // HighMem covers the upper part of the address space.
  uptr max_address = GetMaxUserVirtualAddress();
  if (SHADOW_OFFSET)
    // Adjust max address to make sure that kHighMemEnd and kHighMemStart are
    // properly aligned:
    max_address |= SHADOW_GRANULARITY * GetMmapGranularity() - 1;
  return max_address;
}

static void InitializeShadowBaseAddress(uptr shadow_size_bytes) {
  // Set the shadow memory address to uninitialized.
  __hwasan_shadow_memory_dynamic_address = kDefaultShadowSentinel;
  uptr shadow_start = SHADOW_OFFSET;
  // Detect if a dynamic shadow address must be used and find the available
  // location when necessary. When dynamic address is used, the macro
  // kLowShadowBeg expands to __hwasan_shadow_memory_dynamic_address which
  // was just set to kDefaultShadowSentinel.
  if (shadow_start == kDefaultShadowSentinel) {
    __hwasan_shadow_memory_dynamic_address = 0;
    CHECK_EQ(0, SHADOW_OFFSET);
    shadow_start = FindDynamicShadowStart(shadow_size_bytes);
  }
  // Update the shadow memory address (potentially) used by instrumentation.
  __hwasan_shadow_memory_dynamic_address = shadow_start;
}

bool InitShadow() {
  // Define the entire memory range.
  kHighMemEnd = GetHighMemEnd();

  // Determine shadow memory base offset.
  InitializeShadowBaseAddress(MEM_TO_SHADOW_SIZE(kHighMemEnd));

  // Place the low memory first.
  if (SHADOW_OFFSET) {
    kLowMemEnd = SHADOW_OFFSET - 1;
    kLowMemStart = 0;
  } else {
    // LowMem covers as much of the first 4GB as possible.
    kLowMemEnd = (1UL << 32) - 1;
    kLowMemStart = MEM_TO_SHADOW(kLowMemEnd) + 1;
  }

  // Define the low shadow based on the already placed low memory.
  kLowShadowEnd = MEM_TO_SHADOW(kLowMemEnd);
  kLowShadowStart = SHADOW_OFFSET ? SHADOW_OFFSET : MEM_TO_SHADOW(kLowMemStart);

  // High shadow takes whatever memory is left up there (making sure it is not
  // interfering with low memory in the fixed case).
  kHighShadowEnd = MEM_TO_SHADOW(kHighMemEnd);
  kHighShadowStart = Max(kLowMemEnd, MEM_TO_SHADOW(kHighShadowEnd)) + 1;

  // High memory starts where allocated shadow allows.
  kHighMemStart = SHADOW_TO_MEM(kHighShadowStart);

  // Check the sanity of the defined memory ranges (there might be gaps).
  CHECK_EQ(kHighMemStart % GetMmapGranularity(), 0);
  CHECK_GT(kHighMemStart, kHighShadowEnd);
  CHECK_GT(kHighShadowEnd, kHighShadowStart);
  CHECK_GT(kHighShadowStart, kLowMemEnd);
  CHECK_GT(kLowMemEnd, kLowMemStart);
  CHECK_GT(kLowShadowEnd, kLowShadowStart);
  if (SHADOW_OFFSET)
    CHECK_GT(kLowShadowStart, kLowMemEnd);
  else
    CHECK_GT(kLowMemEnd, kLowShadowStart);

  if (Verbosity())
    PrintAddressSpaceLayout();

  // Reserve shadow memory.
  ReserveShadowMemoryRange(kLowShadowStart, kLowShadowEnd, "low shadow");
  ReserveShadowMemoryRange(kHighShadowStart, kHighShadowEnd, "high shadow");

  // Protect all the gaps.
  ProtectGap(0, Min(kLowMemStart, kLowShadowStart));
  if (SHADOW_OFFSET) {
    if (kLowMemEnd + 1 < kLowShadowStart)
      ProtectGap(kLowMemEnd + 1, kLowShadowStart - kLowMemEnd - 1);
    if (kLowShadowEnd + 1 < kHighShadowStart)
      ProtectGap(kLowShadowEnd + 1, kHighShadowStart - kLowShadowEnd - 1);
  } else {
    if (kLowMemEnd + 1 < kHighShadowStart)
      ProtectGap(kLowMemEnd + 1, kHighShadowStart - kLowMemEnd - 1);
  }
  if (kHighShadowEnd + 1 < kHighMemStart)
    ProtectGap(kHighShadowEnd + 1, kHighMemStart - kHighShadowEnd - 1);

  return true;
}

bool MemIsApp(uptr p) {
  CHECK(GetTagFromPointer(p) == 0);
  return p >= kHighMemStart || (p >= kLowMemStart && p <= kLowMemEnd);
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
  bool recover;
};

static AccessInfo GetAccessInfo(siginfo_t *info, ucontext_t *uc) {
  // Access type is passed in a platform dependent way (see below) and encoded
  // as 0xXY, where X&1 is 1 for store, 0 for load, and X&2 is 1 if the error is
  // recoverable. Valid values of Y are 0 to 4, which are interpreted as
  // log2(access_size), and 0xF, which means that access size is passed via
  // platform dependent register (see below).
#if defined(__aarch64__)
  // Access type is encoded in BRK immediate as 0x900 + 0xXY. For Y == 0xF,
  // access size is stored in X1 register. Access address is always in X0
  // register.
  uptr pc = (uptr)info->si_addr;
  const unsigned code = ((*(u32 *)pc) >> 5) & 0xffff;
  if ((code & 0xff00) != 0x900)
    return AccessInfo{}; // Not ours.

  const bool is_store = code & 0x10;
  const bool recover = code & 0x20;
  const uptr addr = uc->uc_mcontext.regs[0];
  const unsigned size_log = code & 0xf;
  if (size_log > 4 && size_log != 0xf)
    return AccessInfo{}; // Not ours.
  const uptr size = size_log == 0xf ? uc->uc_mcontext.regs[1] : 1U << size_log;

#elif defined(__x86_64__)
  // Access type is encoded in the instruction following INT3 as
  // NOP DWORD ptr [EAX + 0x40 + 0xXY]. For Y == 0xF, access size is stored in
  // RSI register. Access address is always in RDI register.
  uptr pc = (uptr)uc->uc_mcontext.gregs[REG_RIP];
  uint8_t *nop = (uint8_t*)pc;
  if (*nop != 0x0f || *(nop + 1) != 0x1f || *(nop + 2) != 0x40  ||
      *(nop + 3) < 0x40)
    return AccessInfo{}; // Not ours.
  const unsigned code = *(nop + 3);

  const bool is_store = code & 0x10;
  const bool recover = code & 0x20;
  const uptr addr = uc->uc_mcontext.gregs[REG_RDI];
  const unsigned size_log = code & 0xf;
  if (size_log > 4 && size_log != 0xf)
    return AccessInfo{}; // Not ours.
  const uptr size =
      size_log == 0xf ? uc->uc_mcontext.gregs[REG_RSI] : 1U << size_log;

#else
# error Unsupported architecture
#endif

  return AccessInfo{addr, size, is_store, !is_store, recover};
}

static bool HwasanOnSIGTRAP(int signo, siginfo_t *info, ucontext_t *uc) {
  AccessInfo ai = GetAccessInfo(info, uc);
  if (!ai.is_store && !ai.is_load)
    return false;

  InternalMmapVector<BufferedStackTrace> stack_buffer(1);
  BufferedStackTrace *stack = stack_buffer.data();
  stack->Reset();
  SignalContext sig{info, uc};
  GetStackTrace(stack, kStackTraceMax, sig.pc, sig.bp, uc,
                common_flags()->fast_unwind_on_fatal);

  ReportTagMismatch(stack, ai.addr, ai.size, ai.is_store);

  ++hwasan_report_count;
  if (flags()->halt_on_error || !ai.recover)
    Die();

#if defined(__aarch64__)
  uc->uc_mcontext.pc += 4;
#elif defined(__x86_64__)
#else
# error Unsupported architecture
#endif
  return true;
}

static void OnStackUnwind(const SignalContext &sig, const void *,
                          BufferedStackTrace *stack) {
  GetStackTrace(stack, kStackTraceMax, sig.pc, sig.bp, sig.context,
                common_flags()->fast_unwind_on_fatal);
}

void HwasanOnDeadlySignal(int signo, void *info, void *context) {
  // Probably a tag mismatch.
  if (signo == SIGTRAP)
    if (HwasanOnSIGTRAP(signo, (siginfo_t *)info, (ucontext_t*)context))
      return;

  HandleDeadlySignal(info, context, GetTid(), &OnStackUnwind, nullptr);
}


} // namespace __hwasan

#endif // SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD
