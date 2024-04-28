//===-- FPExceptMatchers.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPExceptMatcher.h"

#include "test/UnitTest/Test.h"

#include "hdr/types/fenv_t.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include <memory>
#include <setjmp.h>
#include <signal.h>

#if LIBC_TEST_HAS_MATCHERS()

namespace LIBC_NAMESPACE {
namespace testing {

#if defined(_WIN32)
#define sigjmp_buf jmp_buf
#define sigsetjmp(buf, save) setjmp(buf)
#define siglongjmp(buf, val) longjmp(buf, val)
#endif

static thread_local sigjmp_buf jumpBuffer;
static thread_local bool caughtExcept;

static void sigfpeHandler(int sig) {
  caughtExcept = true;
  siglongjmp(jumpBuffer, -1);
}

FPExceptMatcher::FPExceptMatcher(FunctionCaller *func) {
  auto oldSIGFPEHandler = signal(SIGFPE, &sigfpeHandler);
  std::unique_ptr<FunctionCaller> funcUP(func);

  caughtExcept = false;
  fenv_t oldEnv;
  fputil::get_env(&oldEnv);
  if (sigsetjmp(jumpBuffer, 1) == 0)
    funcUP->call();
  // We restore the previous floating point environment after
  // the call to the function which can potentially raise SIGFPE.
  fputil::set_env(&oldEnv);
  signal(SIGFPE, oldSIGFPEHandler);
  exceptionRaised = caughtExcept;
}

} // namespace testing
} // namespace LIBC_NAMESPACE

#endif // LIBC_TEST_HAS_MATCHERS()
