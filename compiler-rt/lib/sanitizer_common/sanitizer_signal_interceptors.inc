//===-- sanitizer_signal_interceptors.inc -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#if SANITIZER_NETBSD
#define sigaction_symname __sigaction14
#else
#define sigaction_symname sigaction
#endif

#ifndef SIGNAL_INTERCEPTOR_SIGNAL_IMPL
#define SIGNAL_INTERCEPTOR_SIGNAL_IMPL(func, signum, handler) \
  { return REAL(func)(signum, handler); }
#endif

#ifndef SIGNAL_INTERCEPTOR_SIGACTION_IMPL
#  define SIGNAL_INTERCEPTOR_SIGACTION_IMPL(signum, act, oldact)              \
    {                                                                         \
      if (!REAL(sigaction_symname)) {                                         \
        Printf(                                                               \
            "Warning: REAL(sigaction_symname) == nullptr. This may happen "   \
            "if you link with ubsan statically. Sigaction will not work.\n"); \
        return -1;                                                            \
      }                                                                       \
      return REAL(sigaction_symname)(signum, act, oldact);                    \
    }
#endif

#if SANITIZER_INTERCEPT_BSD_SIGNAL
INTERCEPTOR(uptr, bsd_signal, int signum, uptr handler) {
  SIGNAL_INTERCEPTOR_ENTER();
  if (GetHandleSignalMode(signum) == kHandleSignalExclusive) return 0;

  // TODO: support cloak_sanitizer_signal_handlers
  SIGNAL_INTERCEPTOR_SIGNAL_IMPL(bsd_signal, signum, handler);
}
#define INIT_BSD_SIGNAL COMMON_INTERCEPT_FUNCTION(bsd_signal)
#else  // SANITIZER_INTERCEPT_BSD_SIGNAL
#define INIT_BSD_SIGNAL
#endif  // SANITIZER_INTERCEPT_BSD_SIGNAL

#if SANITIZER_INTERCEPT_SIGNAL_AND_SIGACTION
INTERCEPTOR(uptr, signal, int signum, uptr handler) {
  SIGNAL_INTERCEPTOR_ENTER();
  if (GetHandleSignalMode(signum) == kHandleSignalExclusive)
    // The user can neither view nor change the signal handler, regardless of
    // the cloak_sanitizer_signal_handlers setting. This differs from
    // sigaction().
    return (uptr) nullptr;

  uptr ret = +[](auto signal, int signum, uptr handler) {
    SIGNAL_INTERCEPTOR_SIGNAL_IMPL(signal, signum, handler);
  }(signal, signum, handler);

  if (ret != sig_err && SetSignalHandlerFromSanitizer(signum, false))
    // If the user sets a signal handler, it becomes uncloaked, even if they
    // reuse a sanitizer's signal handler.
    ret = sig_dfl;

  return ret;
}
#define INIT_SIGNAL COMMON_INTERCEPT_FUNCTION(signal)

INTERCEPTOR(int, sigaction_symname, int signum,
            const __sanitizer_sigaction *act, __sanitizer_sigaction *oldact) {
  SIGNAL_INTERCEPTOR_ENTER();

  if (GetHandleSignalMode(signum) == kHandleSignalExclusive) {
    if (!oldact) return 0;
    act = nullptr;
    // If cloak_sanitizer_signal_handlers=true, the user can neither view nor
    // change the signal handle.
    // If false, the user can view but not change the signal handler. This
    // differs from signal().
  }

  int ret = +[](int signum, const __sanitizer_sigaction* act,
                __sanitizer_sigaction* oldact) {
    SIGNAL_INTERCEPTOR_SIGACTION_IMPL(signum, act, oldact);
  }(signum, act, oldact);

  if (act) {
    if (ret == 0 && SetSignalHandlerFromSanitizer(signum, false)) {
      // If the user sets a signal handler, it becomes uncloaked, even if they
      // reuse a sanitizer's signal handler.

      if (oldact)
        oldact->handler = reinterpret_cast<__sanitizer_sighandler_ptr>(sig_dfl);
    }
  } else if (ret == 0 && oldact && IsSignalHandlerFromSanitizer(signum)) {
    oldact->handler = reinterpret_cast<__sanitizer_sighandler_ptr>(sig_dfl);
  }

  return ret;
}
#define INIT_SIGACTION COMMON_INTERCEPT_FUNCTION(sigaction_symname)

namespace __sanitizer {
int real_sigaction(int signum, const void *act, void *oldact) {
  return REAL(sigaction_symname)(signum, (const __sanitizer_sigaction *)act,
                         (__sanitizer_sigaction *)oldact);
}
}  // namespace __sanitizer
#else  // SANITIZER_INTERCEPT_SIGNAL_AND_SIGACTION
#define INIT_SIGNAL
#define INIT_SIGACTION
// We need to have defined REAL(sigaction) on other systems.
namespace __sanitizer {
struct __sanitizer_sigaction;
}
DEFINE_REAL(int, sigaction, int signum, const __sanitizer_sigaction *act,
            __sanitizer_sigaction *oldact)
#endif  // SANITIZER_INTERCEPT_SIGNAL_AND_SIGACTION

static void InitializeSignalInterceptors() {
  static bool was_called_once;
  CHECK(!was_called_once);
  was_called_once = true;

  INIT_BSD_SIGNAL;
  INIT_SIGNAL;
  INIT_SIGACTION;
}
