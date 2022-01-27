//===-- Implementation of the sqrtf function for x86_64 -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sqrtf.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, sqrtf, (float x)) {
  float result;
  __asm__ __volatile__("sqrtss %x1, %x0" : "=x"(result) : "x"(x));
  return result;
}

} // namespace __llvm_libc
