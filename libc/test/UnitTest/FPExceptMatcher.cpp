//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// FPExceptMatchers.cpp.
///
//===----------------------------------------------------------------------===//

#include "FPExceptMatcher.h"

#include "src/__support/macros/config.h"
#include "test/UnitTest/ExecuteFunction.h" // FunctionCaller
#include "test/UnitTest/Test.h"

#include "hdr/types/fenv_t.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include <setjmp.h>

// To make this test work on bare-metal targets without working signal.h, find
// out if signal-macros.h did anything, before including the full signal.h. It
// doesn't define a specific macro of the form HAVE_SIGNALS, so we just test
// for one of the macros it _does_ define.
#include "llvm-libc-macros/signal-macros.h"
#ifdef __NSIGSET_WORDS
#include <signal.h>
#endif

#if LIBC_TEST_HAS_MATCHERS()

namespace LIBC_NAMESPACE_DECL {
namespace testing {

#if defined(_WIN32)
#define sigjmp_buf jmp_buf
#define sigsetjmp(buf, save) setjmp(buf)
#define siglongjmp(buf, val) longjmp(buf, val)
#endif

#ifdef __FreeBSD__
using sighandler_t = __sighandler_t *;
#endif

static thread_local bool caughtExcept;

#ifdef __NSIGSET_WORDS

static thread_local sigjmp_buf jumpBuffer;

static void sigfpeHandler([[maybe_unused]] int sig) {
  caughtExcept = true;
  siglongjmp(jumpBuffer, -1);
}

#endif // __NSIGSET_WORDS

FPExceptMatcher::FPExceptMatcher(FunctionCaller *func) {
#ifdef __NSIGSET_WORDS
  auto *oldSIGFPEHandler = signal(SIGFPE, &sigfpeHandler);
#endif

  caughtExcept = false;
  fenv_t oldEnv;
  fputil::get_env(&oldEnv);
#ifdef __NSIGSET_WORDS
  if (sigsetjmp(jumpBuffer, 1) == 0)
#endif
    func->call();
  delete func;
  // We restore the previous floating point environment after
  // the call to the function which can potentially raise SIGFPE.
  fputil::set_env(&oldEnv);
#ifdef __NSIGSET_WORDS
  signal(SIGFPE, oldSIGFPEHandler);
#endif
  exceptionRaised = caughtExcept;
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TEST_HAS_MATCHERS()
