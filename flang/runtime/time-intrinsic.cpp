//===-- runtime/time-intrinsic.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements time-related intrinsic subroutines.

#include "time-intrinsic.h"

#include <ctime>

namespace Fortran::runtime {
extern "C" {

// CPU_TIME (Fortran 2018 16.9.57)
double RTNAME(CpuTime)() {
  // This is part of the c++ standard, so it should at least exist everywhere.
  // It probably does not have the best resolution, so we prefer other
  // platform-specific alternatives if they exist.
  std::clock_t timestamp{std::clock()};
  if (timestamp != std::clock_t{-1}) {
    return static_cast<double>(timestamp) / CLOCKS_PER_SEC;
  }

  // Return some negative value to represent failure.
  return -1.0;
}
} // extern "C"
} // namespace Fortran::runtime
