//===-- Implementation of the sqrtl function for x86_64 -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sqrtl.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long double, sqrtl, (long double x)) {
  long double result;
  __asm__ __volatile__("fsqrt" : "=t"(result) : "t"(x));
  return result;
}

} // namespace __llvm_libc
