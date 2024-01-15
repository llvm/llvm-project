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

// Defines the layout (sign, exponent, significand) of a floating point type in
// memory. It also defines its associated StorageType, i.e., the unsigned
// integer type used to manipulate its representation.
template <FPType> struct FPLayout {};

template <> struct FPLayout<FPType::IEEE754_Binary16> {
  using StorageType = uint16_t;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 5;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 10;
};

template <> struct FPLayout<FPType::IEEE754_Binary32> {
  using StorageType = uint32_t;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 8;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 23;
};

template <> struct FPLayout<FPType::IEEE754_Binary64> {
  using StorageType = uint64_t;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 11;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 52;
};

template <> struct FPLayout<FPType::IEEE754_Binary128> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 112;
};

template <> struct FPLayout<FPType::X86_Binary80> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 64;
};

} // namespace internal

// FPRepBase derives useful constants from the FPLayout.
template <FPType fp_type>
struct FPRepBase : public internal::FPLayout<fp_type> {
private:
  using UP = internal::FPLayout<fp_type>;

public:
  using UP::EXP_LEN;  // The number of bits for the *exponent* part
  using UP::SIG_LEN;  // The number of bits for the *significand* part
  using UP::SIGN_LEN; // The number of bits for the *sign* part
  // For convenience, the sum of `SIG_LEN`, `EXP_LEN`, and `SIGN_LEN`.
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = SIGN_LEN + EXP_LEN + SIG_LEN;

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

  // Merge bits from 'a' and 'b' values according to 'mask'.
  // Use 'a' bits when corresponding 'mask' bits are zeroes and 'b' bits when
  // corresponding bits are ones.
  LIBC_INLINE static constexpr StorageType merge(StorageType a, StorageType b,
                                                 StorageType mask) {
    // https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
    return a ^ ((a ^ b) & mask);
  }

protected:
  // The number of bits after the decimal dot when the number is in normal form.
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN =
      fp_type == FPType::X86_Binary80 ? SIG_LEN - 1 : SIG_LEN;
  LIBC_INLINE_VAR static constexpr uint32_t MANTISSA_PRECISION =
      FRACTION_LEN + 1;
  LIBC_INLINE_VAR static constexpr StorageType FRACTION_MASK =
      mask_trailing_ones<StorageType, FRACTION_LEN>();

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QUIET_NAN_MASK & bits(x) != 0
  LIBC_INLINE_VAR static constexpr StorageType QUIET_NAN_MASK =
      fp_type == FPType::X86_Binary80
          ? bit_at(SIG_LEN - 1) | bit_at(SIG_LEN - 2) // 0b1100...
          : bit_at(SIG_LEN - 1);                      // 0b1000...

  // Mask to generate a default signaling NAN. Any NAN that is not
  // a quiet NAN is considered a signaling NAN.
  LIBC_INLINE_VAR static constexpr StorageType DEFAULT_SIGNALING_NAN =
      fp_type == FPType::X86_Binary80
          ? bit_at(SIG_LEN - 1) | bit_at(SIG_LEN - 3) // 0b1010...
          : bit_at(SIG_LEN - 2);                      // 0b0100...

  // The floating point number representation as an unsigned integer.
  StorageType bits = 0;

public:
  LIBC_INLINE constexpr bool get_sign() const {
    return (bits & SIGN_MASK) != 0;
  }

  LIBC_INLINE constexpr void set_sign(bool signVal) {
    if (get_sign() != signVal)
      bits ^= SIGN_MASK;
  }

  LIBC_INLINE constexpr StorageType get_mantissa() const {
    return bits & FRACTION_MASK;
  }

  LIBC_INLINE constexpr void set_mantissa(StorageType mantVal) {
    bits = merge(bits, mantVal, FRACTION_MASK);
  }

  LIBC_INLINE constexpr uint16_t get_biased_exponent() const {
    return uint16_t((bits & EXP_MASK) >> EXP_MASK_SHIFT);
  }

  LIBC_INLINE constexpr void set_biased_exponent(StorageType biased) {
    bits = merge(bits, biased << EXP_MASK_SHIFT, EXP_MASK);
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

namespace internal {

// Manipulates the representation of a floating point number defined by its
// FPType. This layer is architecture agnostic and does not handle C++ floating
// point types directly ('float', 'double' and 'long double'). Use the FPBits
// below if needed.
//
// TODO: Specialize this class for FPType::X86_Binary80 and remove ad-hoc logic
// from FPRepBase.
template <FPType fp_type> struct FPRep : public FPRepBase<fp_type> {
  using UP = FPRepBase<fp_type>;
  using typename UP::StorageType;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;
  using UP::MANTISSA_PRECISION;
};

} // namespace internal

// Returns the FPType corresponding to C++ type T on the host.
template <typename T> LIBC_INLINE static constexpr FPType get_fp_type() {
  using UnqualT = cpp::remove_cv_t<T>;
  if constexpr (cpp::is_same_v<UnqualT, float> && __FLT_MANT_DIG__ == 24)
    return FPType::IEEE754_Binary32;
  else if constexpr (cpp::is_same_v<UnqualT, double> && __DBL_MANT_DIG__ == 53)
    return FPType::IEEE754_Binary64;
  else if constexpr (cpp::is_same_v<UnqualT, long double>) {
    if constexpr (__LDBL_MANT_DIG__ == 53)
      return FPType::IEEE754_Binary64;
    else if constexpr (__LDBL_MANT_DIG__ == 64)
      return FPType::X86_Binary80;
    else if constexpr (__LDBL_MANT_DIG__ == 113)
      return FPType::IEEE754_Binary128;
  }
#if defined(LIBC_COMPILER_HAS_C23_FLOAT16)
  else if constexpr (cpp::is_same_v<UnqualT, _Float16>)
    return FPType::IEEE754_Binary16;
#endif
#if defined(LIBC_COMPILER_HAS_C23_FLOAT128)
  else if constexpr (cpp::is_same_v<UnqualT, _Float128>)
    return FPType::IEEE754_Binary128;
#endif
#if defined(LIBC_COMPILER_HAS_FLOAT128_EXTENSION)
  else if constexpr (cpp::is_same_v<UnqualT, __float128>)
    return FPType::IEEE754_Binary128;
#endif
  else
    static_assert(cpp::always_false<UnqualT>, "Unsupported type");
}

// A generic class to represent single precision, double precision, and quad
// precision IEEE 754 floating point formats.
// On most platforms, the 'float' type corresponds to single precision floating
// point numbers, the 'double' type corresponds to double precision floating
// point numers, and the 'long double' type corresponds to the quad precision
// floating numbers. On x86 platforms however, the 'long double' type maps to
// an x87 floating point format. This format is an IEEE 754 extension format.
// It is handled as an explicit specialization of this class.
template <typename T> struct FPBits : public internal::FPRep<get_fp_type<T>()> {
  static_assert(cpp::is_floating_point_v<T>,
                "FPBits instantiated with invalid type.");
  using UP = internal::FPRep<get_fp_type<T>()>;

private:
  using UP::EXP_SIG_MASK;
  using UP::QUIET_NAN_MASK;
  using UP::SIG_LEN;
  using UP::SIG_MASK;

public:
  using StorageType = typename UP::StorageType;
  using UP::bits;
  using UP::EXP_BIAS;
  using UP::EXP_LEN;
  using UP::EXP_MASK;
  using UP::EXP_MASK_SHIFT;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;
  using UP::SIGN_MASK;
  using UP::TOTAL_LEN;
  using UP::UP;

  using UP::get_biased_exponent;
  using UP::is_zero;
  // Constants.
  static constexpr int MAX_BIASED_EXPONENT = (1 << EXP_LEN) - 1;
  static constexpr StorageType MIN_SUBNORMAL = StorageType(1);
  static constexpr StorageType MAX_SUBNORMAL = FRACTION_MASK;
  static constexpr StorageType MIN_NORMAL = (StorageType(1) << FRACTION_LEN);
  static constexpr StorageType MAX_NORMAL =
      (StorageType(MAX_BIASED_EXPONENT - 1) << SIG_LEN) | SIG_MASK;

  // Constructors.
  LIBC_INLINE constexpr FPBits() = default;

  template <typename XType> LIBC_INLINE constexpr explicit FPBits(XType x) {
    using Unqual = typename cpp::remove_cv_t<XType>;
    if constexpr (cpp::is_same_v<Unqual, T>) {
      bits = cpp::bit_cast<StorageType>(x);
    } else if constexpr (cpp::is_same_v<Unqual, StorageType>) {
      bits = x;
    } else {
      // We don't want accidental type promotions/conversions, so we require
      // exact type match.
      static_assert(cpp::always_false<XType>);
    }
  }
  // Floating-point conversions.
  LIBC_INLINE constexpr T get_val() const { return cpp::bit_cast<T>(bits); }

  LIBC_INLINE constexpr explicit operator T() const { return get_val(); }

  // The function return mantissa with the implicit bit set iff the current
  // value is a valid normal number.
  LIBC_INLINE constexpr StorageType get_explicit_mantissa() {
    return ((get_biased_exponent() > 0 && !is_inf_or_nan())
                ? (FRACTION_MASK + 1)
                : 0) |
           (FRACTION_MASK & bits);
  }

  LIBC_INLINE constexpr bool is_inf() const {
    return (bits & EXP_SIG_MASK) == EXP_MASK;
  }

  LIBC_INLINE constexpr bool is_nan() const {
    return (bits & EXP_SIG_MASK) > EXP_MASK;
  }

  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return (bits & EXP_SIG_MASK) >= (EXP_MASK | QUIET_NAN_MASK);
  }

  LIBC_INLINE constexpr bool is_inf_or_nan() const {
    return (bits & EXP_MASK) == EXP_MASK;
  }

  LIBC_INLINE constexpr FPBits abs() const {
    return FPBits(bits & EXP_SIG_MASK);
  }

  // Methods below this are used by tests.

  LIBC_INLINE static constexpr T zero(bool sign = false) {
    StorageType rep = (sign ? SIGN_MASK : StorageType(0)) // sign
                      | 0                                 // exponent
                      | 0;                                // mantissa
    return FPBits(rep).get_val();
  }

  LIBC_INLINE static constexpr T neg_zero() { return zero(true); }

  LIBC_INLINE static constexpr T inf(bool sign = false) {
    StorageType rep = (sign ? SIGN_MASK : StorageType(0)) // sign
                      | EXP_MASK                          // exponent
                      | 0;                                // mantissa
    return FPBits(rep).get_val();
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
    StorageType rep = 0                      // sign
                      | EXP_MASK             // exponent
                      | (v & FRACTION_MASK); // mantissa
    return FPBits(rep).get_val();
  }

  LIBC_INLINE static constexpr T build_quiet_nan(StorageType v) {
    return build_nan(QUIET_NAN_MASK | v);
  }

  LIBC_INLINE static constexpr FPBits<T>
  create_value(bool sign, StorageType biased_exp, StorageType mantissa) {
    StorageType rep = (sign ? SIGN_MASK : StorageType(0))           // sign
                      | ((biased_exp << EXP_MASK_SHIFT) & EXP_MASK) // exponent
                      | (mantissa & FRACTION_MASK);                 // mantissa
    return FPBits(rep);
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
};

} // namespace fputil
} // namespace LIBC_NAMESPACE

#ifdef LIBC_LONG_DOUBLE_IS_X86_FLOAT80
#include "x86_64/LongDoubleBits.h"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H
