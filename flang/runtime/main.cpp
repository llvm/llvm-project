//===-- runtime/main.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/main.h"
#include "environment.h"
#include "terminator.h"
#include <cfenv>
#include <cstdio>
#include <cstdlib>

static void ConfigureFloatingPoint() {
#ifdef feclearexcept // a macro in some environments; omit std::
  feclearexcept(FE_ALL_EXCEPT);
#else
  std::feclearexcept(FE_ALL_EXCEPT);
#endif
#ifdef fesetround
  fesetround(FE_TONEAREST);
#else
  std::fesetround(FE_TONEAREST);
#endif
}

extern "C" {
void RTNAME(ProgramStart)(int argc, const char *argv[], const char *envp[],
    const EnvironmentDefaultList *envDefaults) {
  std::atexit(Fortran::runtime::NotifyOtherImagesOfNormalEnd);
  Fortran::runtime::executionEnvironment.Configure(
      argc, argv, envp, envDefaults);
  ConfigureFloatingPoint();
  // I/O is initialized on demand so that it works for non-Fortran main().
}

void RTNAME(ByteswapOption)() {
  if (Fortran::runtime::executionEnvironment.conversion ==
      Fortran::runtime::Convert::Unknown) {
    // The environment variable overrides the command-line option;
    // either of them take precedence over explicit OPEN(CONVERT=) specifiers.
    Fortran::runtime::executionEnvironment.conversion =
        Fortran::runtime::Convert::Swap;
  }
}
}
