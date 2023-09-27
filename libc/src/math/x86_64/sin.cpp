//===-- Implementation of the sin function for x86_64 ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sin.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, sin, (double x)) {
  __asm__ __volatile__("fsin" : "+t"(x));
  return x;
}

} // namespace LIBC_NAMESPACE
