//===-- Comparision operations on floating point numbers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_COMPARISIONOPERATIONS_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_COMPARISIONOPERATIONS_H

#include "FEnvImpl.h"                      // raise_except_if_required
#include "FPBits.h"                        // FPBits<T>
#include "src/__support/CPP/type_traits.h" // enable_if, is_floating_point
#include "src/__support/macros/config.h"   // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

// IEEE Standard 754-2019. Section 5.11
// Rules for comparision within the same floating point type
// 1. +0 = −0
// 2. (i)   +inf  = +inf
//    (ii)  -inf  = -inf
//    (iii) -inf != +inf
// 3. Any comparision with NaN return false except (NaN != NaN => true)
template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<T>, bool> equals(T x,
                                                                       T y) {
  using FPBits = FPBits<T>;
  FPBits x_bits(x);
  FPBits y_bits(y);

  if (x_bits.is_signaling_nan() || y_bits.is_signaling_nan())
    fputil::raise_except_if_required(FE_INVALID);

  // NaN == x returns false for every x
  if (x_bits.is_nan() || y_bits.is_nan())
    return false;

  // +/- 0 == +/- 0
  if (x_bits.is_zero() && y_bits.is_zero())
    return true;

  // should also work for comparisions of different signs
  return x_bits.uintval() == y_bits.uintval();
}

// !(x == y) => x != y
template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<T>, bool>
not_equals(T x, T y) {
  return !equals(x, y);
}

// Rules:
// 1. -inf < x (x != -inf)
// 2. x < +inf (x != +inf)
// 3. Any comparision with NaN return false
template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<T>, bool> less_than(T x,
                                                                          T y) {
  using FPBits = FPBits<T>;
  FPBits x_bits(x);
  FPBits y_bits(y);

  if (x_bits.is_signaling_nan() || y_bits.is_signaling_nan())
    fputil::raise_except_if_required(FE_INVALID);

  // Any comparision with NaN returns false
  if (x_bits.is_nan() || y_bits.is_nan())
    return false;

  if (x_bits.is_zero() && y_bits.is_zero())
    return false;

  if (x_bits.is_neg() && y_bits.is_pos())
    return true;

  if (x_bits.is_pos() && y_bits.is_neg())
    return false;

  // since we store the float in the format: s | e | m
  // the comparisions should work if we directly compare the uintval's

  // TODO: verify if we should use FPBits.get_exponent and FPBits.get_mantissa
  // instead of directly comparing uintval's

  // both negative
  if (x_bits.is_neg())
    return x_bits.uintval() > y_bits.uintval();

  // both positive
  return x_bits.uintval() < y_bits.uintval();
}

// x < y => y > x
template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<T>, bool>
greater_than(T x, T y) {
  return less_than(y, x);
}

// following is expression is correct, accounting for NaN case(s) as well
// x <= y => (x < y) || (x == y)
template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<T>, bool>
less_than_or_equals(T x, T y) {
  return less_than(x, y) || equals(x, y);
}

// x >= y => (x > y) || (x == y) => (y < x) || (x == y)
template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<T>, bool>
greater_than_or_equals(T x, T y) {
  return less_than(y, x) || equals(x, y);
}

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_COMPARISIONOPERATIONS_H
