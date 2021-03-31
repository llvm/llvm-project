//===-- Implementation of the truncf function for aarch64 -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/truncf.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, truncf, (float x)) {
  float y;
  __asm__ __volatile__("frintz %s0, %s1\n\t" : "=w"(y) : "w"(x));
  return y;
}

} // namespace __llvm_libc
