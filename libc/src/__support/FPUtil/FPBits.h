//===-- Abstract class for bit manipulation of float numbers. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE

#include "FloatProperties.h"
#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace fputil {

// A generic class to represent single precision, double precision, and quad
// precision IEEE 754 floating point formats.
// On most platforms, the 'float' type corresponds to single precision floating
// point numbers, the 'double' type corresponds to double precision floating
// point numers, and the 'long double' type corresponds to the quad precision
// floating numbers. On x86 platforms however, the 'long double' type maps to
// an x87 floating point format. This format is an IEEE 754 extension format.
// It is handled as an explicit specialization of this class.
template <typename T> struct FPBits : private FloatProperties<T> {
  static_assert(cpp::is_floating_point_v<T>,
                "FPBits instantiated with invalid type.");
  using typename FloatProperties<T>::StorageType;
  using FloatProperties<T>::TOTAL_LEN;
  using FloatProperties<T>::EXP_MANT_MASK;
  using FloatProperties<T>::EXP_MASK;
  using FloatProperties<T>::EXP_BIAS;
  using FloatProperties<T>::EXP_LEN;
  using FloatProperties<T>::FRACTION_MASK;
  using FloatProperties<T>::FRACTION_LEN;
  using FloatProperties<T>::QUIET_NAN_MASK;
  using FloatProperties<T>::SIGN_MASK;

  // Reinterpreting bits as an integer value and interpreting the bits of an
  // integer value as a floating point value is used in tests. So, a convenient
  // type is provided for such reinterpretations.
  StorageType bits;

  LIBC_INLINE constexpr void set_mantissa(StorageType mantVal) {
    mantVal &= FRACTION_MASK;
    bits &= ~FRACTION_MASK;
    bits |= mantVal;
  }

  LIBC_INLINE constexpr StorageType get_mantissa() const {
    return bits & FRACTION_MASK;
  }

  LIBC_INLINE constexpr void set_biased_exponent(StorageType expVal) {
    expVal = (expVal << FRACTION_LEN) & EXP_MASK;
    bits &= ~EXP_MASK;
    bits |= expVal;
  }

  LIBC_INLINE constexpr uint16_t get_biased_exponent() const {
    return uint16_t((bits & EXP_MASK) >> FRACTION_LEN);
  }

  // The function return mantissa with the implicit bit set iff the current
  // value is a valid normal number.
  LIBC_INLINE constexpr StorageType get_explicit_mantissa() {
    return ((get_biased_exponent() > 0 && !is_inf_or_nan())
                ? (FRACTION_MASK + 1)
                : 0) |
           (FRACTION_MASK & bits);
  }

  LIBC_INLINE constexpr void set_sign(bool signVal) {
    bits |= SIGN_MASK;
    if (!signVal)
      bits -= SIGN_MASK;
  }

  LIBC_INLINE constexpr bool get_sign() const {
    return (bits & SIGN_MASK) != 0;
  }

  static_assert(sizeof(T) == sizeof(StorageType),
                "Data type and integral representation have different sizes.");

  static constexpr int MAX_EXPONENT = (1 << EXP_LEN) - 1;

  static constexpr StorageType MIN_SUBNORMAL = StorageType(1);
  static constexpr StorageType MAX_SUBNORMAL = FRACTION_MASK;
  static constexpr StorageType MIN_NORMAL = (StorageType(1) << FRACTION_LEN);
  static constexpr StorageType MAX_NORMAL =
      ((StorageType(MAX_EXPONENT) - 1) << FRACTION_LEN) | MAX_SUBNORMAL;

  // We don't want accidental type promotions/conversions, so we require exact
  // type match.
  template <typename XType, cpp::enable_if_t<cpp::is_same_v<T, XType>, int> = 0>
  LIBC_INLINE constexpr explicit FPBits(XType x)
      : bits(cpp::bit_cast<StorageType>(x)) {}

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<XType, StorageType>, int> = 0>
  LIBC_INLINE constexpr explicit FPBits(XType x) : bits(x) {}

  LIBC_INLINE constexpr FPBits() : bits(0) {}

  LIBC_INLINE constexpr T get_val() const { return cpp::bit_cast<T>(bits); }

  LIBC_INLINE constexpr void set_val(T value) {
    bits = cpp::bit_cast<StorageType>(value);
  }

  LIBC_INLINE constexpr explicit operator T() const { return get_val(); }

  LIBC_INLINE constexpr StorageType uintval() const { return bits; }

  LIBC_INLINE constexpr int get_exponent() const {
    return int(get_biased_exponent()) - EXP_BIAS;
  }

  // If the number is subnormal, the exponent is treated as if it were the
  // minimum exponent for a normal number. This is to keep continuity between
  // the normal and subnormal ranges, but it causes problems for functions where
  // values are calculated from the exponent, since just subtracting the bias
  // will give a slightly incorrect result. Additionally, zero has an exponent
  // of zero, and that should actually be treated as zero.
  LIBC_INLINE constexpr int get_explicit_exponent() const {
    const int biased_exp = int(get_biased_exponent());
    if (is_zero()) {
      return 0;
    } else if (biased_exp == 0) {
      return 1 - EXP_BIAS;
    } else {
      return biased_exp - EXP_BIAS;
    }
  }

  LIBC_INLINE constexpr bool is_zero() const {
    // Remove sign bit by shift
    return (bits << 1) == 0;
  }

  LIBC_INLINE constexpr bool is_inf() const {
    return (bits & EXP_MANT_MASK) == EXP_MASK;
  }

  LIBC_INLINE constexpr bool is_nan() const {
    return (bits & EXP_MANT_MASK) > EXP_MASK;
  }

  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return (bits & EXP_MANT_MASK) == (EXP_MASK | QUIET_NAN_MASK);
  }

  LIBC_INLINE constexpr bool is_inf_or_nan() const {
    return (bits & EXP_MASK) == EXP_MASK;
  }

  LIBC_INLINE static constexpr T zero(bool sign = false) {
    return FPBits(sign ? SIGN_MASK : StorageType(0)).get_val();
  }

  LIBC_INLINE static constexpr T neg_zero() { return zero(true); }

  LIBC_INLINE static constexpr T inf(bool sign = false) {
    return FPBits((sign ? SIGN_MASK : StorageType(0)) | EXP_MASK).get_val();
  }

  LIBC_INLINE static constexpr T neg_inf() { return inf(true); }

  LIBC_INLINE static constexpr T min_normal() {
    return FPBits(MIN_NORMAL).get_val();
  }

  LIBC_INLINE static constexpr T max_normal() {
    return FPBits(MAX_NORMAL).get_val();
  }

  LIBC_INLINE static constexpr T min_denormal() {
    return FPBits(MIN_SUBNORMAL).get_val();
  }

  LIBC_INLINE static constexpr T max_denormal() {
    return FPBits(MAX_SUBNORMAL).get_val();
  }

  LIBC_INLINE static constexpr T build_nan(StorageType v) {
    FPBits<T> bits(inf());
    bits.set_mantissa(v);
    return T(bits);
  }

  LIBC_INLINE static constexpr T build_quiet_nan(StorageType v) {
    return build_nan(QUIET_NAN_MASK | v);
  }

  // The function convert integer number and unbiased exponent to proper float
  // T type:
  //   Result = number * 2^(ep+1 - exponent_bias)
  // Be careful!
  //   1) "ep" is raw exponent value.
  //   2) The function add to +1 to ep for seamless normalized to denormalized
  //      transition.
  //   3) The function did not check exponent high limit.
  //   4) "number" zero value is not processed correctly.
  //   5) Number is unsigned, so the result can be only positive.
  LIBC_INLINE static constexpr FPBits<T> make_value(StorageType number,
                                                    int ep) {
    FPBits<T> result;
    // offset: +1 for sign, but -1 for implicit first bit
    int lz = cpp::countl_zero(number) - EXP_LEN;
    number <<= lz;
    ep -= lz;

    if (LIBC_LIKELY(ep >= 0)) {
      // Implicit number bit will be removed by mask
      result.set_mantissa(number);
      result.set_biased_exponent(ep + 1);
    } else {
      result.set_mantissa(number >> -ep);
    }
    return result;
  }

  LIBC_INLINE static constexpr FPBits<T>
  create_value(bool sign, StorageType biased_exp, StorageType mantissa) {
    FPBits<T> result;
    result.set_sign(sign);
    result.set_biased_exponent(biased_exp);
    result.set_mantissa(mantissa);
    return result;
  }
};

} // namespace fputil
} // namespace LIBC_NAMESPACE

#ifdef LIBC_LONG_DOUBLE_IS_X86_FLOAT80
#include "x86_64/LongDoubleBits.h"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H
