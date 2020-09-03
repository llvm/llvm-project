//===-- runtime/environment.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "environment.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace Fortran::runtime {
ExecutionEnvironment executionEnvironment;

void ExecutionEnvironment::Configure(
    int ac, const char *av[], const char *env[]) {
  argc = ac;
  argv = av;
  envp = env;
  listDirectedOutputLineLengthLimit = 79; // PGI default
  defaultOutputRoundingMode =
      decimal::FortranRounding::RoundNearest; // RP(==RN)

  if (auto *x{std::getenv("FORT_FMT_RECL")}) {
    char *end;
    auto n{std::strtol(x, &end, 10)};
    if (n > 0 && n < std::numeric_limits<int>::max() && *end == '\0') {
      listDirectedOutputLineLengthLimit = n;
    } else {
      std::fprintf(
          stderr, "Fortran runtime: FORT_FMT_RECL=%s is invalid; ignored\n", x);
    }
  }

  // TODO: Set RP/ROUND='PROCESSOR_DEFINED' from environment
}
} // namespace Fortran::runtime
