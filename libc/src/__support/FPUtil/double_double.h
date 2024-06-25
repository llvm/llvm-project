//===-- Utilities for double-double data type. ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_DOUBLE_DOUBLE_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_DOUBLE_DOUBLE_H

#include "multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA
#include "src/__support/number_pair.h"

namespace LIBC_NAMESPACE::fputil {

using DoubleDouble = LIBC_NAMESPACE::NumberPair<double>;

// The output of Dekker's FastTwoSum algorithm is correct, i.e.:
//   r.hi + r.lo = a + b exactly
//   and |r.lo| < eps(r.lo)
// Assumption: |a| >= |b|, or a = 0.
template <bool FAST2SUM = true>
LIBC_INLINE constexpr DoubleDouble exact_add(double a, double b) {
  DoubleDouble r{0.0, 0.0};
  if constexpr (FAST2SUM) {
    r.hi = a + b;
    double t = r.hi - a;
    r.lo = b - t;
  } else {
    r.hi = a + b;
    double t1 = r.hi - a;
    double t2 = r.hi - t1;
    double t3 = b - t1;
    double t4 = a - t2;
    r.lo = t3 + t4;
  }
  return r;
}

// Assumption: |a.hi| >= |b.hi|
LIBC_INLINE constexpr DoubleDouble add(const DoubleDouble &a,
                                       const DoubleDouble &b) {
  DoubleDouble r = exact_add(a.hi, b.hi);
  double lo = a.lo + b.lo;
  return exact_add(r.hi, r.lo + lo);
}

// Assumption: |a.hi| >= |b|
LIBC_INLINE constexpr DoubleDouble add(const DoubleDouble &a, double b) {
  DoubleDouble r = exact_add<false>(a.hi, b);
  return exact_add(r.hi, r.lo + a.lo);
}

// Veltkamp's Splitting for double precision.
// Note: This is proved to be correct for all rounding modes:
//   Zimmermann, P., "Note on the Veltkamp/Dekker Algorithms with Directed
//   Roundings," https://inria.hal.science/hal-04480440.
// Default splitting constant = 2^ceil(prec(double)/2) + 1 = 2^27 + 1.
template <size_t N = 27> LIBC_INLINE constexpr DoubleDouble split(double a) {
  DoubleDouble r{0.0, 0.0};
  // CN = 2^N.
  constexpr double CN = static_cast<double>(1 << N);
  constexpr double C = CN + 1.0;
  double t1 = C * a;
  double t2 = a - t1;
  r.hi = t1 + t2;
  r.lo = a - r.hi;
  return r;
}

// Note: When FMA instruction is not available, the `exact_mult` function is
// only correct for round-to-nearest mode.  See:
//   Zimmermann, P., "Note on the Veltkamp/Dekker Algorithms with Directed
//   Roundings," https://inria.hal.science/hal-04480440.
// Using Theorem 1 in the paper above, without FMA instruction, if we restrict
// the generated constants to precision <= 51, and splitting it by 2^28 + 1,
// then a * b = r.hi + r.lo is exact for all rounding modes.
template <bool NO_FMA_ALL_ROUNDINGS = false>
LIBC_INLINE DoubleDouble exact_mult(double a, double b) {
  DoubleDouble r{0.0, 0.0};

#ifdef LIBC_TARGET_CPU_HAS_FMA
  r.hi = a * b;
  r.lo = fputil::multiply_add(a, b, -r.hi);
#else
  // Dekker's Product.
  DoubleDouble as = split(a);
  DoubleDouble bs;

  if constexpr (NO_FMA_ALL_ROUNDINGS)
    bs = split<28>(b);
  else
    bs = split(b);

  r.hi = a * b;
  double t1 = as.hi * bs.hi - r.hi;
  double t2 = as.hi * bs.lo + t1;
  double t3 = as.lo * bs.hi + t2;
  r.lo = as.lo * bs.lo + t3;
#endif // LIBC_TARGET_CPU_HAS_FMA

  return r;
}

LIBC_INLINE DoubleDouble quick_mult(double a, const DoubleDouble &b) {
  DoubleDouble r = exact_mult(a, b.hi);
  r.lo = multiply_add(a, b.lo, r.lo);
  return r;
}

template <bool NO_FMA_ALL_ROUNDINGS = false>
LIBC_INLINE DoubleDouble quick_mult(const DoubleDouble &a,
                                    const DoubleDouble &b) {
  DoubleDouble r = exact_mult<NO_FMA_ALL_ROUNDINGS>(a.hi, b.hi);
  double t1 = multiply_add(a.hi, b.lo, r.lo);
  double t2 = multiply_add(a.lo, b.hi, t1);
  r.lo = t2;
  return r;
}

// Assuming |c| >= |a * b|.
template <>
LIBC_INLINE DoubleDouble multiply_add<DoubleDouble>(const DoubleDouble &a,
                                                    const DoubleDouble &b,
                                                    const DoubleDouble &c) {
  return add(c, quick_mult(a, b));
}

} // namespace LIBC_NAMESPACE::fputil

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_DOUBLE_DOUBLE_H
