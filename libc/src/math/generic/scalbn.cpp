//===-- Single-precision scalbn function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalbn.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, scalbn, (double x, int n)) {
#if !defined(__FLT_RADIX__)
#error __FLT_RADIX__ undefined.
#elif __FLT_RADIX__ != 2
#error __FLT_RADIX__!=2, unimplemented.
#else
  return fputil::ldexp(x, n);
#endif
}

} // namespace __llvm_libc
