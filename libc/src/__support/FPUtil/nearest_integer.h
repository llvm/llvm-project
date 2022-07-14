//===-- Fast rounding to nearest integer for floating point -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_NEAREST_INTEGER_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_NEAREST_INTEGER_H

#include "src/__support/architectures.h"
#include "src/__support/common.h"

#if (defined(LLVM_LIBC_ARCH_X86_64) && defined(__SSE4_2__))
#include "x86_64/nearest_integer.h"
#elif defined(LLVM_LIBC_ARCH_AARCH64)
#include "aarch64/nearest_integer.h"
#else

namespace __llvm_libc {
namespace fputil {

// This is a fast implementation for rounding to a nearest integer that, in case
// of a tie, might pick a random one among 2 closest integers when the rounding
// mode is not FE_TONEAREST.
//
// Notice that for AARCH64 and x86-64 with SSE4.2 support, we will use their
// corresponding rounding instruction instead.  And in those cases, the results
// are rounded to the nearest integer, tie-to-even.
static inline double nearest_integer(double x) {
  if (x < 0x1p53 && x > -0x1p53) {
    double r = x < 0 ? (x - 0x1.0p52) + 0x1.0p52 : (x + 0x1.0p52) - 0x1.0p52;
    double diff = x - r;
    // The expression above is correct for the default rounding mode, round-to-
    // nearest, tie-to-even.  For other rounding modes, it might be off by 1,
    // which is corrected below.
    if (unlikely(diff > 0.5))
      return r + 1.0;
    if (unlikely(diff < -0.5))
      return r - 1.0;
    return r;
  }
  return x;
}

} // namespace fputil
} // namespace __llvm_libc

#endif
#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_NEAREST_INTEGER_H
