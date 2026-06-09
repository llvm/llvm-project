//===-- Check exceptions for exp functions ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_CHECK_EXP_EXCEPTIONS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_CHECK_EXP_EXCEPTIONS_H

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace check {

namespace exp_internal {

template <typename T> struct Bounds;

template <> struct Bounds<float> {
  // Smallest value that will cause overflow, generated from Sollya:
  // > float_max = 2^127 * (2 - 2^-23);
  // > upper = round(log(float_max), SG, RU);
  // > printfloat(upper);
  static constexpr uint32_t UPPER = 0x42b1'7218;
  // Largest value that will cause underflow, generated from Sollya:
  // > float_min = 2^-126;
  // > lower = round(log(float_min), SG, RD);
  // > printfloat(lower);
  static constexpr uint32_t LOWER = 0xc2ae'ac50;
};

template <> struct Bounds<double> {
  // Smallest value that will cause overflow, generated from Sollya:
  // > double_max = 2^1023 * (2 - 2^-52);
  // > upper = round(log(double_max), D, RU);
  // > printdouble(upper);
  static constexpr uint64_t UPPER = 0x4086'2e42'fefa'39f0;
  // Largest value that will cause underflow, generated from Sollya:
  // > double_min = 2^-1023;
  // > lower = round(log(double_min), D, RD);
  // > printfloat(lower);
  static constexpr uint64_t LOWER = 0xc086'232b'dd7a'bcd3;
};

} // namespace exp_internal

template <typename T>
LIBC_INLINE unsigned exp_exceptions(T x, unsigned rounding_mode) {
  using FPBits = typename fputil::FPBits<T>;
  FPBits x_bits(x);
  if (x_bits.is_signaling_nan())
    return FE_INVALID;
  if (x_bits.is_inf_or_nan() || x_bits.is_zero())
    return 0;
  FPbits::StorageType x_u = x_bits.uintval();
  if (x_u >= exp_internal::Bounds<T>::UPPER && x_bits.is_pos())
    return (rounding_mode == FE_TONEAREST || rounding_mode == FE_UPWARD)
               ? (FE_OVERFLOW | FE_INEXACT)
               : FE_INEXACT;
  if (x_u >= exp_internal::Bounds<T>::LOWER)
    return FE_UNDERFLOW | FE_INEXACT;
  return FE_INEXACT;
}

} // namespace check

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_CHECK_EXP_EXCEPTIONS_H
