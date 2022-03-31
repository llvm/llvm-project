//===-- Abstract class for bit manipulation of float numbers. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_H

#include "PlatformDefs.h"

#include "src/__support/CPP/Bit.h"
#include "src/__support/CPP/TypeTraits.h"

#include "FloatProperties.h"
#include <stdint.h>

namespace __llvm_libc {
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
  static_assert(cpp::IsFloatingPointType<T>::Value,
                "FPBits instantiated with invalid type.");

  // Reinterpreting bits as an integer value and interpreting the bits of an
  // integer value as a floating point value is used in tests. So, a convenient
  // type is provided for such reinterpretations.
  using FloatProp = FloatProperties<T>;
  // TODO: Change UintType name to BitsType for consistency.
  using UIntType = typename FloatProp::BitsType;

  UIntType bits;

  void set_mantissa(UIntType mantVal) {
    mantVal &= (FloatProp::MANTISSA_MASK);
    bits &= ~(FloatProp::MANTISSA_MASK);
    bits |= mantVal;
  }

  UIntType get_mantissa() const { return bits & FloatProp::MANTISSA_MASK; }

  void set_unbiased_exponent(UIntType expVal) {
    expVal = (expVal << (FloatProp::MANTISSA_WIDTH)) & FloatProp::EXPONENT_MASK;
    bits &= ~(FloatProp::EXPONENT_MASK);
    bits |= expVal;
  }

  uint16_t get_unbiased_exponent() const {
    return uint16_t((bits & FloatProp::EXPONENT_MASK) >>
                    (FloatProp::MANTISSA_WIDTH));
  }

  void set_sign(bool signVal) {
    bits &= ~(FloatProp::SIGN_MASK);
    UIntType sign = UIntType(signVal) << (FloatProp::BIT_WIDTH - 1);
    bits |= sign;
  }

  bool get_sign() const {
    return ((bits & FloatProp::SIGN_MASK) >> (FloatProp::BIT_WIDTH - 1));
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

  // We don't want accidental type promotions/conversions so we require exact
  // type match.
  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<T, XType>::Value, int> = 0>
  constexpr explicit FPBits(XType x)
      : bits(__llvm_libc::bit_cast<UIntType>(x)) {}

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<XType, UIntType>::Value, int> = 0>
  constexpr explicit FPBits(XType x) : bits(x) {}

  FPBits() : bits(0) {}

  T get_val() const { return __llvm_libc::bit_cast<T>(bits); }

  void set_val(T value) { bits = __llvm_libc::bit_cast<UIntType>(value); }

  explicit operator T() const { return get_val(); }

  UIntType uintval() const { return bits; }

  int get_exponent() const {
    return int(get_unbiased_exponent()) - EXPONENT_BIAS;
  }

  bool is_zero() const {
    return get_mantissa() == 0 && get_unbiased_exponent() == 0;
  }

  bool is_inf() const {
    return get_mantissa() == 0 && get_unbiased_exponent() == MAX_EXPONENT;
  }

  bool is_nan() const {
    return get_unbiased_exponent() == MAX_EXPONENT && get_mantissa() != 0;
  }

  bool is_inf_or_nan() const { return get_unbiased_exponent() == MAX_EXPONENT; }

  static FPBits<T> zero() { return FPBits(); }

  static FPBits<T> neg_zero() {
    return FPBits(UIntType(1) << (sizeof(UIntType) * 8 - 1));
  }

  static FPBits<T> inf() {
    FPBits<T> bits;
    bits.set_unbiased_exponent(MAX_EXPONENT);
    return bits;
  }

  static FPBits<T> neg_inf() {
    FPBits<T> bits = inf();
    bits.set_sign(1);
    return bits;
  }

  static T build_nan(UIntType v) {
    FPBits<T> bits = inf();
    bits.set_mantissa(v);
    return T(bits);
  }
};

} // namespace fputil
} // namespace __llvm_libc

#ifdef SPECIAL_X86_LONG_DOUBLE
#include "x86_64/LongDoubleBits.h"
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_H
