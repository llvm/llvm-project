//===-- asan_posix.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Posix-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "asan_internal.h"
#include "asan_interceptors.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_procmaps.h"

#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

namespace __asan {

void AsanOnDeadlySignal(int signo, void *siginfo, void *context) {
  ScopedDeadlySignal signal_scope(GetCurrentThread());
  int code = (int)((siginfo_t*)siginfo)->si_code;
  // Write the first message using fd=2, just in case.
  // It may actually fail to write in case stderr is closed.
  internal_write(2, "ASAN:DEADLYSIGNAL\n", 18);
  SignalContext sig = SignalContext::Create(siginfo, context);

  // Access at a reasonable offset above SP, or slightly below it (to account
  // for x86_64 or PowerPC redzone, ARM push of multiple registers, etc) is
  // probably a stack overflow.
#ifdef __s390__
  // On s390, the fault address in siginfo points to start of the page, not
  // to the precise word that was accessed.  Mask off the low bits of sp to
  // take it into account.
  bool IsStackAccess = sig.addr >= (sig.sp & ~0xFFF) &&
                       sig.addr < sig.sp + 0xFFFF;
#else
  bool IsStackAccess = sig.addr + 512 > sig.sp && sig.addr < sig.sp + 0xFFFF;
#endif

#if __powerpc__
  // Large stack frames can be allocated with e.g.
  //   lis r0,-10000
  //   stdux r1,r1,r0 # store sp to [sp-10000] and update sp by -10000
  // If the store faults then sp will not have been updated, so test above
  // will not work, becase the fault address will be more than just "slightly"
  // below sp.
  if (!IsStackAccess && IsAccessibleMemoryRange(sig.pc, 4)) {
    u32 inst = *(unsigned *)sig.pc;
    u32 ra = (inst >> 16) & 0x1F;
    u32 opcd = inst >> 26;
    u32 xo = (inst >> 1) & 0x3FF;
    // Check for store-with-update to sp. The instructions we accept are:
    //   stbu rs,d(ra)          stbux rs,ra,rb
    //   sthu rs,d(ra)          sthux rs,ra,rb
    //   stwu rs,d(ra)          stwux rs,ra,rb
    //   stdu rs,ds(ra)         stdux rs,ra,rb
    // where ra is r1 (the stack pointer).
    if (ra == 1 &&
        (opcd == 39 || opcd == 45 || opcd == 37 || opcd == 62 ||
         (opcd == 31 && (xo == 247 || xo == 439 || xo == 183 || xo == 181))))
      IsStackAccess = true;
  }
#endif // __powerpc__

  // We also check si_code to filter out SEGV caused by something else other
  // then hitting the guard page or unmapped memory, like, for example,
  // unaligned memory access.
  if (IsStackAccess && (code == si_SEGV_MAPERR || code == si_SEGV_ACCERR))
    ReportStackOverflow(sig);
  else
    ReportDeadlySignal(signo, sig);
}

// ---------------------- TSD ---------------- {{{1

static pthread_key_t tsd_key;
static bool tsd_key_inited = false;
void AsanTSDInit(void (*destructor)(void *tsd)) {
  CHECK(!tsd_key_inited);
  tsd_key_inited = true;
  CHECK_EQ(0, pthread_key_create(&tsd_key, destructor));
}

void *AsanTSDGet() {
  CHECK(tsd_key_inited);
  return pthread_getspecific(tsd_key);
}

void AsanTSDSet(void *tsd) {
  CHECK(tsd_key_inited);
  pthread_setspecific(tsd_key, tsd);
}

void PlatformTSDDtor(void *tsd) {
  AsanThreadContext *context = (AsanThreadContext*)tsd;
  if (context->destructor_iterations > 1) {
    context->destructor_iterations--;
    CHECK_EQ(0, pthread_setspecific(tsd_key, tsd));
    return;
  }
  AsanThread::TSDDtor(tsd);
}
}  // namespace __asan

#endif  // SANITIZER_POSIX
