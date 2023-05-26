//===-- Utilities for double-double data type. ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_DOUBLEDOUBLE_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_DOUBLEDOUBLE_H

#include "multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA
#include "src/__support/number_pair.h"

namespace __llvm_libc::fputil {

using DoubleDouble = __llvm_libc::NumberPair<double>;

// The output of Dekker's FastTwoSum algorithm is correct, i.e.:
//   r.hi + r.lo = a + b exactly
//   and |r.lo| < eps(r.lo)
// if ssumption: |a| >= |b|, or a = 0.
LIBC_INLINE constexpr DoubleDouble exact_add(double a, double b) {
  DoubleDouble r{0.0, 0.0};
  r.hi = a + b;
  double t = r.hi - a;
  r.lo = b - t;
  return r;
}

// Assumption: |a.hi| >= |b.hi|
LIBC_INLINE constexpr DoubleDouble add(DoubleDouble a, DoubleDouble b) {
  DoubleDouble r = exact_add(a.hi, b.hi);
  double lo = a.lo + b.lo;
  return exact_add(r.hi, r.lo + lo);
}

// Assumption: |a.hi| >= |b|
LIBC_INLINE constexpr DoubleDouble add(DoubleDouble a, double b) {
  DoubleDouble r = exact_add(a.hi, b);
  return exact_add(r.hi, r.lo + a.lo);
}

// Velkamp's Splitting for double precision.
LIBC_INLINE constexpr DoubleDouble split(double a) {
  DoubleDouble r{0.0, 0.0};
  // Splitting constant = 2^ceil(prec(double)/2) + 1 = 2^27 + 1.
  constexpr double C = 0x1.0p27 + 1.0;
  double t1 = C * a;
  double t2 = a - t1;
  r.hi = t1 + t2;
  r.lo = a - r.hi;
  return r;
}

LIBC_INLINE DoubleDouble exact_mult(double a, double b) {
  DoubleDouble r{0.0, 0.0};

#ifdef LIBC_TARGET_CPU_HAS_FMA
  r.hi = a * b;
  r.lo = fputil::multiply_add(a, b, -r.hi);
#else
  // Dekker's Product.
  DoubleDouble as = split(a);
  DoubleDouble bs = split(b);
  r.hi = a * b;
  double t1 = as.hi * bs.hi - r.hi;
  double t2 = as.hi * bs.lo + t1;
  double t3 = as.lo * bs.hi + t2;
  r.lo = as.lo * bs.lo + t3;
#endif // LIBC_TARGET_CPU_HAS_FMA

  return r;
}

LIBC_INLINE DoubleDouble quick_mult(DoubleDouble a, DoubleDouble b) {
  DoubleDouble r = exact_mult(a.hi, b.hi);
  double t1 = fputil::multiply_add(a.hi, b.lo, r.lo);
  double t2 = fputil::multiply_add(a.lo, b.hi, t1);
  r.lo = t2;
  return r;
}

} // namespace __llvm_libc::fputil

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_DOUBLEDOUBLE_H
