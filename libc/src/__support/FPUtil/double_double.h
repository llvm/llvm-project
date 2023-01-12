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
#include "src/__support/number_pair.h"

namespace __llvm_libc::fputil {

using DoubleDouble = __llvm_libc::NumberPair<double>;

// Assumption: |a| >= |b|
constexpr inline DoubleDouble exact_add(double a, double b) {
  DoubleDouble r{0.0, 0.0};
  r.hi = a + b;
  double t = r.hi - a;
  r.lo = b - t;
  return r;
}

// Assumption: |a.hi| >= |b.hi|
constexpr inline DoubleDouble add(DoubleDouble a, DoubleDouble b) {
  DoubleDouble r = exact_add(a.hi, b.hi);
  double lo = a.lo + b.lo;
  return exact_add(r.hi, r.lo + lo);
}

// Assumption: |a.hi| >= |b|
constexpr inline DoubleDouble add(DoubleDouble a, double b) {
  DoubleDouble r = exact_add(a.hi, b);
  return exact_add(r.hi, r.lo + a.lo);
}

// TODO(lntue): add a correct multiplication when FMA instructions are not
// available.
inline DoubleDouble exact_mult(double a, double b) {
  DoubleDouble r{0.0, 0.0};
  r.hi = a * b;
  r.lo = fputil::multiply_add(a, b, -r.hi);
  return r;
}

} // namespace __llvm_libc::fputil

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_DOUBLEDOUBLE_H
