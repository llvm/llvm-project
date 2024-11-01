//===-- Implementation of the trunc function for aarch64 ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/trunc.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, trunc, (double x)) {
  double y;
  __asm__ __volatile__("frintz %d0, %d1\n" : "=w"(y) : "w"(x));
  return y;
}

} // namespace LIBC_NAMESPACE
