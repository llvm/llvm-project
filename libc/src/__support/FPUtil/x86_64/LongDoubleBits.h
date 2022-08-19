//===-- Bit representation of x86 long double numbers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_LONG_DOUBLE_BITS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_LONG_DOUBLE_BITS_H

#include "src/__support/CPP/Bit.h"
#include "src/__support/CPP/UInt128.h"
#include "src/__support/architectures.h"

#if !defined(LLVM_LIBC_ARCH_X86)
#error "Invalid include"
#endif

#include "src/__support/FPUtil/FPBits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

template <unsigned Width> struct Padding;

// i386 padding.
template <> struct Padding<4> { static constexpr unsigned VALUE = 16; };

// x86_64 padding.
template <> struct Padding<8> { static constexpr unsigned VALUE = 48; };

template <> struct FPBits<long double> {
  using UIntType = UInt128;

  static constexpr int EXPONENT_BIAS = 0x3FFF;
  static constexpr int MAX_EXPONENT = 0x7FFF;
  static constexpr UIntType MIN_SUBNORMAL = UIntType(1);
  // Subnormal numbers include the implicit bit in x86 long double formats.
  static constexpr UIntType MAX_SUBNORMAL =
      (UIntType(1) << (MantissaWidth<long double>::VALUE)) - 1;
  static constexpr UIntType MIN_NORMAL =
      (UIntType(3) << MantissaWidth<long double>::VALUE);
  static constexpr UIntType MAX_NORMAL =
      ((UIntType(MAX_EXPONENT) - 1)
       << (MantissaWidth<long double>::VALUE + 1)) |
      (UIntType(1) << MantissaWidth<long double>::VALUE) | MAX_SUBNORMAL;

  using FloatProp = FloatProperties<long double>;

  UIntType bits;

  void set_mantissa(UIntType mantVal) {
    mantVal &= (FloatProp::MANTISSA_MASK);
    bits &= ~(FloatProp::MANTISSA_MASK);
    bits |= mantVal;
  }

  UIntType get_mantissa() const { return bits & FloatProp::MANTISSA_MASK; }

  UIntType get_explicit_mantissa() const {
    return bits & (FloatProp::MANTISSA_MASK | FloatProp::EXPLICIT_BIT_MASK);
  }

  void set_unbiased_exponent(UIntType expVal) {
    expVal =
        (expVal << (FloatProp::BIT_WIDTH - 1 - FloatProp::EXPONENT_WIDTH)) &
        FloatProp::EXPONENT_MASK;
    bits &= ~(FloatProp::EXPONENT_MASK);
    bits |= expVal;
  }

  uint16_t get_unbiased_exponent() const {
    return uint16_t((bits & FloatProp::EXPONENT_MASK) >>
                    (FloatProp::BIT_WIDTH - 1 - FloatProp::EXPONENT_WIDTH));
  }

  void set_implicit_bit(bool implicitVal) {
    bits &= ~(UIntType(1) << FloatProp::MANTISSA_WIDTH);
    bits |= (UIntType(implicitVal) << FloatProp::MANTISSA_WIDTH);
  }

  bool get_implicit_bit() const {
    return ((bits & (UIntType(1) << FloatProp::MANTISSA_WIDTH)) >>
            FloatProp::MANTISSA_WIDTH);
  }

  void set_sign(bool signVal) {
    bits &= ~(FloatProp::SIGN_MASK);
    UIntType sign1 = UIntType(signVal) << (FloatProp::BIT_WIDTH - 1);
    bits |= sign1;
  }

  bool get_sign() const {
    return ((bits & FloatProp::SIGN_MASK) >> (FloatProp::BIT_WIDTH - 1));
  }

  FPBits() : bits(0) {}

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<long double, XType>, int> = 0>
  explicit FPBits(XType x) : bits(__llvm_libc::bit_cast<UIntType>(x)) {
    // bits starts uninitialized, and setting it to a long double only
    // overwrites the first 80 bits. This clears those upper bits.
    bits = bits & ((UIntType(1) << 80) - 1);
  }

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<XType, UIntType>, int> = 0>
  explicit FPBits(XType x) : bits(x) {}

  operator long double() { return __llvm_libc::bit_cast<long double>(bits); }

  UIntType uintval() {
    // We zero the padding bits as they can contain garbage.
    static constexpr UIntType MASK =
        (UIntType(1) << (sizeof(long double) * 8 -
                         Padding<sizeof(uintptr_t)>::VALUE)) -
        1;
    return bits & MASK;
  }

  int get_exponent() const {
    if (get_unbiased_exponent() == 0)
      return int(1) - EXPONENT_BIAS;
    return int(get_unbiased_exponent()) - EXPONENT_BIAS;
  }

  bool is_zero() const {
    return get_unbiased_exponent() == 0 && get_mantissa() == 0 &&
           get_implicit_bit() == 0;
  }

  bool is_inf() const {
    return get_unbiased_exponent() == MAX_EXPONENT && get_mantissa() == 0 &&
           get_implicit_bit() == 1;
  }

  bool is_nan() const {
    if (get_unbiased_exponent() == MAX_EXPONENT) {
      return (get_implicit_bit() == 0) || get_mantissa() != 0;
    } else if (get_unbiased_exponent() != 0) {
      return get_implicit_bit() == 0;
    }
    return false;
  }

  bool is_inf_or_nan() const {
    return (get_unbiased_exponent() == MAX_EXPONENT) ||
           (get_unbiased_exponent() != 0 && get_implicit_bit() == 0);
  }

  // Methods below this are used by tests.

  static FPBits<long double> zero() { return FPBits<long double>(0.0l); }

  static FPBits<long double> neg_zero() {
    FPBits<long double> bits(0.0l);
    bits.set_sign(1);
    return bits;
  }

  static FPBits<long double> inf() {
    FPBits<long double> bits(0.0l);
    bits.set_unbiased_exponent(MAX_EXPONENT);
    bits.set_implicit_bit(1);
    return bits;
  }

  static FPBits<long double> neg_inf() {
    FPBits<long double> bits(0.0l);
    bits.set_unbiased_exponent(MAX_EXPONENT);
    bits.set_implicit_bit(1);
    bits.set_sign(1);
    return bits;
  }

  static long double build_nan(UIntType v) {
    FPBits<long double> bits(0.0l);
    bits.set_unbiased_exponent(MAX_EXPONENT);
    bits.set_implicit_bit(1);
    bits.set_mantissa(v);
    return bits;
  }
};

static_assert(
    sizeof(FPBits<long double>) == sizeof(long double),
    "Internal long double representation does not match the machine format.");

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_LONG_DOUBLE_BITS_H
