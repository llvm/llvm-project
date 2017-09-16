//===-- sanitizer_signal_interceptors.inc -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Signal interceptors for sanitizers.
//
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_platform_interceptors.h"

using namespace __sanitizer;

#if SANITIZER_INTERCEPT_BSD_SIGNAL
INTERCEPTOR(void *, bsd_signal, int signum, void *handler) {
  if (GetHandleSignalMode(signum) == kHandleSignalExclusive) return 0;
  return REAL(bsd_signal)(signum, handler);
}
#define INIT_BSD_SIGNAL COMMON_INTERCEPT_FUNCTION(bsd_signal)
#else  // SANITIZER_INTERCEPT_BSD_SIGNAL
#define INIT_BSD_SIGNAL
#endif  // SANITIZER_INTERCEPT_BSD_SIGNAL

#if SANITIZER_INTERCEPT_SIGNAL_AND_SIGACTION
INTERCEPTOR(void *, signal, int signum, void *handler) {
  if (GetHandleSignalMode(signum) == kHandleSignalExclusive) return nullptr;
  return REAL(signal)(signum, handler);
}
#define INIT_SIGNAL COMMON_INTERCEPT_FUNCTION(signal)

INTERCEPTOR(int, sigaction, int signum, const struct sigaction *act,
            struct sigaction *oldact) {
  if (GetHandleSignalMode(signum) == kHandleSignalExclusive) return 0;
  return REAL(sigaction)(signum, act, oldact);
}
#define INIT_SIGACTION COMMON_INTERCEPT_FUNCTION(sigaction)

namespace __sanitizer {
int real_sigaction(int signum, const void *act, void *oldact) {
  return REAL(sigaction)(signum, (const struct sigaction *)act,
                         (struct sigaction *)oldact);
}
}  // namespace __sanitizer
#else  // SANITIZER_INTERCEPT_SIGNAL_AND_SIGACTION
#define INIT_SIGNAL
#define INIT_SIGACTION
// We need to have defined REAL(sigaction) on other systems.
DEFINE_REAL(int, sigaction, int signum, const struct sigaction *act,
            struct sigaction *oldact)
#endif  // SANITIZER_INTERCEPT_SIGNAL_AND_SIGACTION

static void InitializeSignalInterceptors() {
  static bool was_called_once;
  CHECK(!was_called_once);
  was_called_once = true;

  INIT_BSD_SIGNAL;
  INIT_SIGNAL;
  INIT_SIGACTION;
}
