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
#include "src/__support/UInt128.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE, LIBC_INLINE_VAR
#include "src/__support/macros/properties/float.h" // LIBC_COMPILER_HAS_FLOAT128
#include "src/__support/math_extras.h"             // mask_trailing_ones

#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace fputil {

// The supported floating point types.
enum class FPType {
  IEEE754_Binary16,
  IEEE754_Binary32,
  IEEE754_Binary64,
  IEEE754_Binary128,
  X86_Binary80,
};

namespace internal {

// The type of encoding for supported floating point types.
enum class FPEncoding {
  IEEE754,
  X86_ExtendedPrecision,
};

template <FPType> struct FPBaseProperties {};

template <> struct FPBaseProperties<FPType::IEEE754_Binary16> {
  using StorageType = uint16_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 16;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 10;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 5;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary32> {
  using StorageType = uint32_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 32;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 23;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 8;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary64> {
  using StorageType = uint64_t;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 64;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 52;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 11;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::IEEE754_Binary128> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 128;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 112;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr auto ENCODING = FPEncoding::IEEE754;
};

template <> struct FPBaseProperties<FPType::X86_Binary80> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = 80;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 64;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr auto ENCODING =
      FPEncoding::X86_ExtendedPrecision;
};

} // namespace internal

template <FPType fp_type>
struct FPProperties : public internal::FPBaseProperties<fp_type> {
private:
  using UP = internal::FPBaseProperties<fp_type>;

public:
  // The number of bits to represent sign. For documentation purpose, always 1.
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  using UP::EXP_LEN;   // The number of bits for the *exponent* part
  using UP::SIG_LEN;   // The number of bits for the *significand* part
  using UP::TOTAL_LEN; // For convenience, the sum of `SIG_LEN`, `EXP_LEN`,
                       // and `SIGN_LEN`.
  static_assert(SIGN_LEN + EXP_LEN + SIG_LEN == TOTAL_LEN);

  // An unsigned integer that is wide enough to contain all of the floating
  // point bits.
  using StorageType = typename UP::StorageType;

  // The number of bits in StorageType.
  LIBC_INLINE_VAR static constexpr int STORAGE_LEN =
      sizeof(StorageType) * CHAR_BIT;
  static_assert(STORAGE_LEN >= TOTAL_LEN);

  // The exponent bias. Always positive.
  LIBC_INLINE_VAR static constexpr int32_t EXP_BIAS =
      (1U << (EXP_LEN - 1U)) - 1U;
  static_assert(EXP_BIAS > 0);

protected:
  // The shift amount to get the *significand* part to the least significant
  // bit. Always `0` but kept for consistency.
  LIBC_INLINE_VAR static constexpr int SIG_MASK_SHIFT = 0;
  // The shift amount to get the *exponent* part to the least significant bit.
  LIBC_INLINE_VAR static constexpr int EXP_MASK_SHIFT = SIG_LEN;
  // The shift amount to get the *sign* part to the least significant bit.
  LIBC_INLINE_VAR static constexpr int SIGN_MASK_SHIFT = SIG_LEN + EXP_LEN;

  // The bit pattern that keeps only the *significand* part.
  LIBC_INLINE_VAR static constexpr StorageType SIG_MASK =
      mask_trailing_ones<StorageType, SIG_LEN>() << SIG_MASK_SHIFT;

public:
  // The bit pattern that keeps only the *exponent* part.
  LIBC_INLINE_VAR static constexpr StorageType EXP_MASK =
      mask_trailing_ones<StorageType, EXP_LEN>() << EXP_MASK_SHIFT;
  // The bit pattern that keeps only the *sign* part.
  LIBC_INLINE_VAR static constexpr StorageType SIGN_MASK =
      mask_trailing_ones<StorageType, SIGN_LEN>() << SIGN_MASK_SHIFT;
  // The bit pattern that keeps only the *exponent + significand* part.
  LIBC_INLINE_VAR static constexpr StorageType EXP_SIG_MASK =
      mask_trailing_ones<StorageType, EXP_LEN + SIG_LEN>();
  // The bit pattern that keeps only the *sign + exponent + significand* part.
  LIBC_INLINE_VAR static constexpr StorageType FP_MASK =
      mask_trailing_ones<StorageType, TOTAL_LEN>();

  static_assert((SIG_MASK & EXP_MASK & SIGN_MASK) == 0, "masks disjoint");
  static_assert((SIG_MASK | EXP_MASK | SIGN_MASK) == FP_MASK, "masks cover");

private:
  LIBC_INLINE static constexpr StorageType bit_at(int position) {
    return StorageType(1) << position;
  }

public:
  // The number of bits after the decimal dot when the number is in normal form.
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision ? SIG_LEN - 1
                                                                  : SIG_LEN;
  LIBC_INLINE_VAR static constexpr uint32_t MANTISSA_PRECISION =
      FRACTION_LEN + 1;
  LIBC_INLINE_VAR static constexpr StorageType FRACTION_MASK =
      mask_trailing_ones<StorageType, FRACTION_LEN>();

protected:
  // If a number x is a NAN, then it is a quiet NAN if:
  //   QUIET_NAN_MASK & bits(x) != 0
  LIBC_INLINE_VAR static constexpr StorageType QUIET_NAN_MASK =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision
          ? bit_at(SIG_LEN - 1) | bit_at(SIG_LEN - 2) // 0b1100...
          : bit_at(SIG_LEN - 1);                      // 0b1000...

  // If a number x is a NAN, then it is a signalling NAN if:
  //   SIGNALING_NAN_MASK & bits(x) != 0
  LIBC_INLINE_VAR static constexpr StorageType SIGNALING_NAN_MASK =
      UP::ENCODING == internal::FPEncoding::X86_ExtendedPrecision
          ? bit_at(SIG_LEN - 1) | bit_at(SIG_LEN - 3) // 0b1010...
          : bit_at(SIG_LEN - 2);                      // 0b0100...
};

//-----------------------------------------------------------------------------
template <typename FP> LIBC_INLINE static constexpr FPType get_fp_type() {
  if constexpr (cpp::is_same_v<FP, float> && __FLT_MANT_DIG__ == 24)
    return FPType::IEEE754_Binary32;
  else if constexpr (cpp::is_same_v<FP, double> && __DBL_MANT_DIG__ == 53)
    return FPType::IEEE754_Binary64;
  else if constexpr (cpp::is_same_v<FP, long double>) {
    if constexpr (__LDBL_MANT_DIG__ == 53)
      return FPType::IEEE754_Binary64;
    else if constexpr (__LDBL_MANT_DIG__ == 64)
      return FPType::X86_Binary80;
    else if constexpr (__LDBL_MANT_DIG__ == 113)
      return FPType::IEEE754_Binary128;
  }
#if defined(LIBC_COMPILER_HAS_C23_FLOAT16)
  else if constexpr (cpp::is_same_v<FP, _Float16>)
    return FPType::IEEE754_Binary16;
#endif
#if defined(LIBC_COMPILER_HAS_C23_FLOAT128)
  else if constexpr (cpp::is_same_v<FP, _Float128>)
    return FPType::IEEE754_Binary128;
#endif
#if defined(LIBC_COMPILER_HAS_FLOAT128_EXTENSION)
  else if constexpr (cpp::is_same_v<FP, __float128>)
    return FPType::IEEE754_Binary128;
#endif
  else
    static_assert(cpp::always_false<FP>, "Unsupported type");
}

template <typename FP>
struct FloatProperties : public FPProperties<get_fp_type<FP>()> {};

namespace internal {

// This is a temporary class to unify common methods and properties between
// FPBits and FPBits<long double>.
template <FPType fp_type> struct FPBitsCommon : private FPProperties<fp_type> {
  using UP = FPProperties<fp_type>;
  using typename UP::StorageType;
  using UP::TOTAL_LEN;

protected:
  using UP::EXP_SIG_MASK;
  using UP::QUIET_NAN_MASK;

public:
  using UP::EXP_BIAS;
  using UP::EXP_LEN;
  using UP::EXP_MASK;
  using UP::EXP_MASK_SHIFT;
  using UP::FP_MASK;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;
  using UP::SIGN_MASK;

  // Reinterpreting bits as an integer value and interpreting the bits of an
  // integer value as a floating point value is used in tests. So, a convenient
  // type is provided for such reinterpretations.
  StorageType bits;

  LIBC_INLINE constexpr FPBitsCommon() : bits(0) {}
  LIBC_INLINE explicit constexpr FPBitsCommon(StorageType bits) : bits(bits) {}

  LIBC_INLINE constexpr void set_mantissa(StorageType mantVal) {
    mantVal &= FRACTION_MASK;
    bits &= ~FRACTION_MASK;
    bits |= mantVal;
  }

  LIBC_INLINE constexpr StorageType get_mantissa() const {
    return bits & FRACTION_MASK;
  }

  LIBC_INLINE constexpr void set_sign(bool signVal) {
    if (get_sign() != signVal)
      bits ^= SIGN_MASK;
  }

  LIBC_INLINE constexpr bool get_sign() const {
    return (bits & SIGN_MASK) != 0;
  }

  LIBC_INLINE constexpr void set_biased_exponent(StorageType biased) {
    // clear exponent bits
    bits &= ~EXP_MASK;
    // set exponent bits
    bits |= (biased << EXP_MASK_SHIFT) & EXP_MASK;
  }

  LIBC_INLINE constexpr uint16_t get_biased_exponent() const {
    return uint16_t((bits & EXP_MASK) >> EXP_MASK_SHIFT);
  }

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

  LIBC_INLINE constexpr StorageType uintval() const { return bits & FP_MASK; }

  LIBC_INLINE constexpr bool is_zero() const {
    return (bits & EXP_SIG_MASK) == 0;
  }
};

} // namespace internal

// A generic class to represent single precision, double precision, and quad
// precision IEEE 754 floating point formats.
// On most platforms, the 'float' type corresponds to single precision floating
// point numbers, the 'double' type corresponds to double precision floating
// point numers, and the 'long double' type corresponds to the quad precision
// floating numbers. On x86 platforms however, the 'long double' type maps to
// an x87 floating point format. This format is an IEEE 754 extension format.
// It is handled as an explicit specialization of this class.
template <typename T>
struct FPBits : public internal::FPBitsCommon<get_fp_type<T>()> {
  static_assert(cpp::is_floating_point_v<T>,
                "FPBits instantiated with invalid type.");
  using UP = internal::FPBitsCommon<get_fp_type<T>()>;
  using StorageType = typename UP::StorageType;
  using UP::bits;

private:
  using UP::EXP_SIG_MASK;
  using UP::QUIET_NAN_MASK;

public:
  using UP::EXP_BIAS;
  using UP::EXP_LEN;
  using UP::EXP_MASK;
  using UP::EXP_MASK_SHIFT;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;
  using UP::SIGN_MASK;
  using UP::TOTAL_LEN;

  using UP::get_biased_exponent;
  using UP::is_zero;

  // The function return mantissa with the implicit bit set iff the current
  // value is a valid normal number.
  LIBC_INLINE constexpr StorageType get_explicit_mantissa() {
    return ((get_biased_exponent() > 0 && !is_inf_or_nan())
                ? (FRACTION_MASK + 1)
                : 0) |
           (FRACTION_MASK & bits);
  }

  static constexpr int MAX_BIASED_EXPONENT = (1 << EXP_LEN) - 1;
  static constexpr StorageType MIN_SUBNORMAL = StorageType(1);
  static constexpr StorageType MAX_SUBNORMAL = FRACTION_MASK;
  static constexpr StorageType MIN_NORMAL = (StorageType(1) << FRACTION_LEN);
  static constexpr StorageType MAX_NORMAL =
      ((StorageType(MAX_BIASED_EXPONENT) - 1) << FRACTION_LEN) | MAX_SUBNORMAL;

  // We don't want accidental type promotions/conversions, so we require exact
  // type match.
  template <typename XType, cpp::enable_if_t<cpp::is_same_v<T, XType>, int> = 0>
  LIBC_INLINE constexpr explicit FPBits(XType x)
      : UP(cpp::bit_cast<StorageType>(x)) {}

  template <typename XType,
            cpp::enable_if_t<cpp::is_same_v<XType, StorageType>, int> = 0>
  LIBC_INLINE constexpr explicit FPBits(XType x) : UP(x) {}

  LIBC_INLINE constexpr FPBits() : UP() {}

  LIBC_INLINE constexpr void set_val(T value) {
    bits = cpp::bit_cast<StorageType>(value);
  }

  LIBC_INLINE constexpr T get_val() const { return cpp::bit_cast<T>(bits); }

  LIBC_INLINE constexpr explicit operator T() const { return get_val(); }

  LIBC_INLINE constexpr bool is_inf() const {
    return (bits & EXP_SIG_MASK) == EXP_MASK;
  }

  LIBC_INLINE constexpr bool is_nan() const {
    return (bits & EXP_SIG_MASK) > EXP_MASK;
  }

  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return (bits & EXP_SIG_MASK) == (EXP_MASK | QUIET_NAN_MASK);
  }

  LIBC_INLINE constexpr bool is_inf_or_nan() const {
    return (bits & EXP_MASK) == EXP_MASK;
  }

  LIBC_INLINE constexpr FPBits abs() const {
    return FPBits(bits & EXP_SIG_MASK);
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
