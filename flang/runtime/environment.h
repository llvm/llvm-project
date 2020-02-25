//===-- runtime/environment.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ENVIRONMENT_H_
#define FORTRAN_RUNTIME_ENVIRONMENT_H_

#include "flang/Decimal/decimal.h"

namespace Fortran::runtime {
struct ExecutionEnvironment {
  void Configure(int argc, const char *argv[], const char *envp[]);

  int argc;
  const char **argv;
  const char **envp;
  int listDirectedOutputLineLengthLimit;
  enum decimal::FortranRounding defaultOutputRoundingMode;
};
extern ExecutionEnvironment executionEnvironment;
}

#endif  // FORTRAN_RUNTIME_ENVIRONMENT_H_
