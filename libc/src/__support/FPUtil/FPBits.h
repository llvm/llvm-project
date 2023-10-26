//===-- Abstract class for bit manipulation of float numbers. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H

#include "PlatformDefs.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/builtin_wrappers.h"
#include "src/__support/common.h"

#include "FloatProperties.h"
#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace fputil {

template <typename T> struct MantissaWidth {
  static constexpr unsigned VALUE = FloatProperties<T>::MANTISSA_WIDTH;
};

template <typename T> struct ExponentWidth {
  static constexpr unsigned VALUE = FloatProperties<T>::EXPONENT_WIDTH;
};

// A generic class to represent single precision, double precision, and quad
// precision IEEE 754 floating point formats.
// On most platforms, the 'float' type corresponds to single precision floating
// point numbers, the 'double' type corresponds to double precision floating
// point numers, and the 'long double' type corresponds to the quad precision
// floating numbers. On x86 platforms however, the 'long double' type maps to
// an x87 floating point format. This format is an IEEE 754 extension format.
// It is handled as an explicit specialization of this class.
template <typename T> struct FPBits {
  static_assert(cpp::is_floating_point_v<T>,
                "FPBits instantiated with invalid type.");

  // Reinterpreting bits as an integer value and interpreting the bits of an
  // integer value as a floating point value is used in tests. So, a convenient
  // type is provided for such reinterpretations.
  using FloatProp = FloatProperties<T>;
  // TODO: Change UintType name to BitsType for consistency.
  using UIntType = typename FloatProp::BitsType;

  UIntType bits;

  LIBC_INLINE constexpr void set_mantissa(UIntType mantVal) {
    mantVal &= (FloatProp::MANTISSA_MASK);
    bits &= ~(FloatProp::MANTISSA_MASK);
    bits |= mantVal;
  }

  LIBC_INLINE constexpr UIntType get_mantissa() const {
    return bits & FloatProp::MANTISSA_MASK;
  }

  LIBC_INLINE constexpr void set_unbiased_exponent(UIntType expVal) {
    expVal = (expVal << (FloatProp::MANTISSA_WIDTH)) & FloatProp::EXPONENT_MASK;
    bits &= ~(FloatProp::EXPONENT_MASK);
    bits |= expVal;
  }

  LIBC_INLINE constexpr uint16_t get_unbiased_exponent() const {
    return uint16_t((bits & FloatProp::EXPONENT_MASK) >>
                    (FloatProp::MANTISSA_WIDTH));
  }

  // The function return mantissa with the implicit bit set iff the current
  // value is a valid normal number.
  LIBC_INLINE constexpr UIntType get_explicit_mantissa() {
    return ((get_unbiased_exponent() > 0 && !is_inf_or_nan())
                ? (FloatProp::MANTISSA_MASK + 1)
                : 0) |
           (FloatProp::MANTISSA_MASK & bits);
  }

  LIBC_INLINE constexpr void set_sign(bool signVal) {
    bits |= FloatProp::SIGN_MASK;
    if (!signVal)
      bits -= FloatProp::SIGN_MASK;
  }

  LIBC_INLINE constexpr bool get_sign() const {
    return (bits & FloatProp::SIGN_MASK) != 0;
  }

  static_assert(sizeof(T) == sizeof(UIntType),
                "Data type and integral representation have different sizes.");

  static constexpr int EXPONENT_BIAS = (1 << (ExponentWidth<T>::VALUE - 1)) - 1;
  static constexpr int MAX_EXPONENT = (1 << ExponentWidth<T>::VALUE) - 1;

  static constexpr UIntType MIN_SUBNORMAL = UIntType(1);
  static constexpr UIntType MAX_SUBNORMAL =
      (UIntType(1) << MantissaWidth<T>::VALUE) - 1;
  static constexpr UIntType MIN_NORMAL =
      (UIntType(1) << MantissaWidth<T>::VALUE);
  static constexpr UIntType MAX_NORMAL =
      ((UIntType(MAX_EXPONENT) - 1) << MantissaWidth<T>::VALUE) | MAX_SUBNORMAL;

  // We don't want accidental type promotions/conversions, so we require exact
  // type match.
  template <typename XType, cpp::enable_if_t<cpp::is_same_v<T, XType>, int> = 0>
  constexpr explicit FPBits(XType x) : bits(cpp::bit_cast<UIntType>(x)) {}

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<XType, UIntType>, int> = 0>
  constexpr explicit FPBits(XType x) : bits(x) {}

  FPBits() : bits(0) {}

  LIBC_INLINE constexpr T get_val() const { return cpp::bit_cast<T>(bits); }

  LIBC_INLINE constexpr void set_val(T value) {
    bits = cpp::bit_cast<UIntType>(value);
  }

  LIBC_INLINE constexpr explicit operator T() const { return get_val(); }

  LIBC_INLINE constexpr UIntType uintval() const { return bits; }

  LIBC_INLINE constexpr int get_exponent() const {
    return int(get_unbiased_exponent()) - EXPONENT_BIAS;
  }

  // If the number is subnormal, the exponent is treated as if it were the
  // minimum exponent for a normal number. This is to keep continuity between
  // the normal and subnormal ranges, but it causes problems for functions where
  // values are calculated from the exponent, since just subtracting the bias
  // will give a slightly incorrect result. Additionally, zero has an exponent
  // of zero, and that should actually be treated as zero.
  LIBC_INLINE constexpr int get_explicit_exponent() const {
    const int unbiased_exp = int(get_unbiased_exponent());
    if (is_zero()) {
      return 0;
    } else if (unbiased_exp == 0) {
      return 1 - EXPONENT_BIAS;
    } else {
      return unbiased_exp - EXPONENT_BIAS;
    }
  }

  LIBC_INLINE constexpr bool is_zero() const {
    // Remove sign bit by shift
    return (bits << 1) == 0;
  }

  LIBC_INLINE constexpr bool is_inf() const {
    return (bits & FloatProp::EXP_MANT_MASK) == FloatProp::EXPONENT_MASK;
  }

  LIBC_INLINE constexpr bool is_nan() const {
    return (bits & FloatProp::EXP_MANT_MASK) > FloatProp::EXPONENT_MASK;
  }

  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return (bits & FloatProp::EXP_MANT_MASK) ==
           (FloatProp::EXPONENT_MASK | FloatProp::QUIET_NAN_MASK);
  }

  LIBC_INLINE constexpr bool is_inf_or_nan() const {
    return (bits & FloatProp::EXPONENT_MASK) == FloatProp::EXPONENT_MASK;
  }

  LIBC_INLINE static constexpr FPBits<T> zero(bool sign = false) {
    return FPBits(sign ? FloatProp::SIGN_MASK : UIntType(0));
  }

  LIBC_INLINE static constexpr FPBits<T> neg_zero() { return zero(true); }

  LIBC_INLINE static constexpr FPBits<T> inf(bool sign = false) {
    FPBits<T> bits(sign ? FloatProp::SIGN_MASK : UIntType(0));
    bits.set_unbiased_exponent(MAX_EXPONENT);
    return bits;
  }

  LIBC_INLINE static constexpr FPBits<T> neg_inf() {
    FPBits<T> bits = inf();
    bits.set_sign(1);
    return bits;
  }

  LIBC_INLINE static constexpr FPBits<T> min_normal() {
    return FPBits<T>(MIN_NORMAL);
  }

  LIBC_INLINE static constexpr T build_nan(UIntType v) {
    FPBits<T> bits = inf();
    bits.set_mantissa(v);
    return T(bits);
  }

  LIBC_INLINE static constexpr T build_quiet_nan(UIntType v) {
    return build_nan(FloatProp::QUIET_NAN_MASK | v);
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
  LIBC_INLINE static constexpr FPBits<T> make_value(UIntType number, int ep) {
    FPBits<T> result;
    // offset: +1 for sign, but -1 for implicit first bit
    int lz = unsafe_clz(number) - FloatProp::EXPONENT_WIDTH;
    number <<= lz;
    ep -= lz;

    if (LIBC_LIKELY(ep >= 0)) {
      // Implicit number bit will be removed by mask
      result.set_mantissa(number);
      result.set_unbiased_exponent(ep + 1);
    } else {
      result.set_mantissa(number >> -ep);
    }
    return result;
  }

  LIBC_INLINE static constexpr FPBits<T>
  create_value(bool sign, UIntType unbiased_exp, UIntType mantissa) {
    FPBits<T> result;
    result.set_sign(sign);
    result.set_unbiased_exponent(unbiased_exp);
    result.set_mantissa(mantissa);
    return result;
  }
};

} // namespace fputil
} // namespace LIBC_NAMESPACE

#ifdef SPECIAL_X86_LONG_DOUBLE
#include "x86_64/LongDoubleBits.h"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H
