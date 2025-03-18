//===-- include/flang/Testing/fp-testing.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TESTING_FP_TESTING_H_
#define FORTRAN_TESTING_FP_TESTING_H_

#include "flang/Common/target-rounding.h"
#include <fenv.h>

using Fortran::common::RealFlags;
using Fortran::common::Rounding;
using Fortran::common::RoundingMode;

class ScopedHostFloatingPointEnvironment {
public:
  ScopedHostFloatingPointEnvironment(bool treatSubnormalOperandsAsZero = false,
      bool flushSubnormalResultsToZero = false);
  ~ScopedHostFloatingPointEnvironment();
  void ClearFlags() const;
  static RealFlags CurrentFlags();
  static void SetRounding(Rounding rounding);

private:
  fenv_t originalFenv_;
#if __x86_64__
  unsigned int originalMxcsr;
#endif
};

#endif // FORTRAN_TESTING_FP_TESTING_H_
