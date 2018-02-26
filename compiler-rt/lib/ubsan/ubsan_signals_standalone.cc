//=-- ubsan_signals_standalone.cc
//------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Installs signal handlers and related interceptors for UBSan standalone.
//
//===----------------------------------------------------------------------===//

#include "ubsan_platform.h"
#if CAN_SANITIZE_UB
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "ubsan_diag.h"
#include "ubsan_init.h"

#define COMMON_INTERCEPT_FUNCTION(name) INTERCEPT_FUNCTION(name)
#include "sanitizer_common/sanitizer_signal_interceptors.inc"

namespace __ubsan {

#if SANITIZER_FUCHSIA
void InitializeDeadlySignals() {}
#else
static void OnStackUnwind(const SignalContext &sig, const void *,
                          BufferedStackTrace *stack) {
  GetStackTrace(stack, kStackTraceMax, sig.pc, sig.bp, sig.context,
                common_flags()->fast_unwind_on_fatal);
}

static void UBsanOnDeadlySignal(int signo, void *siginfo, void *context) {
  HandleDeadlySignal(siginfo, context, GetTid(), &OnStackUnwind, nullptr);
}

static bool is_initialized = false;

void InitializeDeadlySignals() {
  if (is_initialized)
    return;
  is_initialized = true;
  InitializeSignalInterceptors();
  InstallDeadlySignalHandlers(&UBsanOnDeadlySignal);
}
#endif

} // namespace __ubsan

#endif // CAN_SANITIZE_UB
