//===-- FPExceptMatchers.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPExceptMatcher.h"

#include <fenv.h>
#include <memory>
#include <setjmp.h>
#include <signal.h>

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
  fegetenv(&oldEnv);
  if (sigsetjmp(jumpBuffer, 1) == 0)
    funcUP->call();
  // We restore the previous floating point environment after
  // the call to the function which can potentially raise SIGFPE.
  fesetenv(&oldEnv);
  signal(SIGFPE, oldSIGFPEHandler);
  exceptionRaised = caughtExcept;
}

} // namespace testing
} // namespace LIBC_NAMESPACE
