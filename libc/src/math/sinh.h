//===-- Implementation header for sinh --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_SINH_H
#define LLVM_LIBC_SRC_MATH_SINH_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/expm1.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE {

double sinh(double x) {
  fputil::FPBits<double> xbits(x);
  uint64_t xbits_u = xbits.bits;
  uint32_t xpt = (xbits_u >> 52) & 0x7ffU;

  // sinh(±Inf) = ±Inf, sinh(NaN) = NaN
  if (xpt >= 0x7ff) {
    if (xbits_u == (0x7ffULL << 52)) // Inf
      return x;
    return x + 0.0; // NaN
  }

  // |x| < 2^-9: sinh(x) ≈ x + x^3/6
  if (xpt < 0x3c9) {
    double x2 = x * x;
    return x + x * x2 * (1.0 / 6.0);
  }

  // Large |x|: sinh(x) ≈ sign(x)*0.5*exp(|x|)
  double h = x < 0.0 ? -x : x;
  double t = expm1(h);
  double s = 0.5 * (t + 1.0);
  return x >= 0.0 ? s : -s;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_SINH_H
