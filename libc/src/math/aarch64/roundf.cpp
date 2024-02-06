//===-- Implementation of the roundf function for aarch64 -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/roundf.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, roundf, (float x)) {
  float y;
  __asm__ __volatile__("frinta %s0, %s1\n\t" : "=w"(y) : "w"(x));
  return y;
}

} // namespace LIBC_NAMESPACE
