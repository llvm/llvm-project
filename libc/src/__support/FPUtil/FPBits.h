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

// The classes hierarchy is as follows:
//
//             ┌───────────────────┐
//             │ FPLayout<FPType>  │
//             └─────────▲─────────┘
//                       │
//             ┌─────────┴─────────┐
//             │ FPRepBase<FPType> │
//             └─────────▲─────────┘
//                       │
//          ┌────────────┴─────────────┐
//          │                          │
// ┌────────┴──────┐     ┌─────────────┴──────────────┐
// │ FPRep<FPType> │     │ FPRep<FPType::X86_Binary80 │
// └────────▲──────┘     └─────────────▲──────────────┘
//          │                          │
//          └────────────┬─────────────┘
//                       │
//                 ┌─────┴─────┐
//                 │ FPBits<T> │
//                 └───────────┘
//
// - 'FPLayout' defines only a few constants, namely the 'StorageType' and the
// length of the sign, the exponent and significand parts.
// - 'FPRepBase' builds more constants on top of those from 'FPLayout' like
// exponent bias, shifts and masks. It also defines tools to assemble or test
// these parts.
// - 'FPRep' defines functions to interact with the floating point
// representation. The default implementation is the one for 'IEEE754', a
// specialization is provided for X86 Extended Precision that has a different
// encoding.
// - 'FPBits' is templated on the platform floating point types. Contrary to
// 'FPRep' that is platform agnostic 'FPBits' is architecture dependent.

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

protected:
  LIBC_INLINE static constexpr StorageType bit_at(int position) {
    return StorageType(1) << position;
  }

  // A stongly typed integer that prevents mixing and matching integers with
  // different semantics.
  template <typename T> struct TypedInt {
    using value_type = T;
    LIBC_INLINE constexpr explicit TypedInt(T value) : value(value) {}
    LIBC_INLINE constexpr TypedInt(const TypedInt &value) = default;

    LIBC_INLINE constexpr explicit operator T() const { return value; }

    LIBC_INLINE constexpr StorageType to_storage_type() const {
      return StorageType(value);
    }

  private:
    T value;
  };

  // An opaque type to store a floating point exponent.
  // We define special values but it is valid to create arbitrary values as long
  // as they are in the range [MIN, MAX].
  struct Exponent : public TypedInt<int32_t> {
    using UP = TypedInt<int32_t>;
    using UP::UP;
    LIBC_INLINE
    static constexpr auto MIN() { return Exponent(1 - EXP_BIAS); }
    LIBC_INLINE static constexpr auto ZERO() { return Exponent(0); }
    LIBC_INLINE static constexpr auto MAX() { return Exponent(EXP_BIAS); }
  };

  // An opaque type to store a floating point biased exponent.
  // We define special values but it is valid to create arbitrary values as long
  // as they are in the range [BITS_ALL_ZEROES, BITS_ALL_ONES].
  // Values greater than BITS_ALL_ONES are truncated.
  struct BiasedExponent : public TypedInt<uint32_t> {
    using UP = TypedInt<uint32_t>;
    using UP::UP;

    LIBC_INLINE constexpr BiasedExponent(Exponent exp)
        : UP(static_cast<int32_t>(exp) + EXP_BIAS) {}
    // The exponent value for denormal numbers.
    LIBC_INLINE static constexpr auto BITS_ALL_ZEROES() {
      return BiasedExponent(uint32_t(0));
    }
    // The exponent value for infinity.
    LIBC_INLINE static constexpr auto BITS_ALL_ONES() {
      return BiasedExponent(uint32_t(2 * EXP_BIAS + 1));
    }
  };

  // An opaque type to store a floating point significand.
  // We define special values but it is valid to create arbitrary values as long
  // as they are in the range [BITS_ALL_ZEROES, BITS_ALL_ONES].
  // Note that the semantics of the Significand are implementation dependent.
  // Values greater than BITS_ALL_ONES are truncated.
  struct Significand : public TypedInt<StorageType> {
    using UP = TypedInt<StorageType>;
    using UP::UP;

    LIBC_INLINE friend constexpr Significand operator|(const Significand a,
                                                       const Significand b) {
      return Significand(
          StorageType(a.to_storage_type() | b.to_storage_type()));
    }
    LIBC_INLINE friend constexpr Significand operator^(const Significand a,
                                                       const Significand b) {
      return Significand(
          StorageType(a.to_storage_type() ^ b.to_storage_type()));
    }
    LIBC_INLINE friend constexpr Significand operator>>(const Significand a,
                                                        int shift) {
      return Significand(StorageType(a.to_storage_type() >> shift));
    }

    LIBC_INLINE static constexpr auto ZERO() {
      return Significand(StorageType(0));
    }
    LIBC_INLINE static constexpr auto LSB() {
      return Significand(StorageType(1));
    }
    LIBC_INLINE static constexpr auto MSB() {
      return Significand(StorageType(bit_at(SIG_LEN - 1)));
    }
    // Aliases
    LIBC_INLINE static constexpr auto BITS_ALL_ZEROES() { return ZERO(); }
    LIBC_INLINE static constexpr auto BITS_ALL_ONES() {
      return Significand(SIG_MASK);
    }
  };

  LIBC_INLINE static constexpr StorageType encode(BiasedExponent exp) {
    return (exp.to_storage_type() << SIG_LEN) & EXP_MASK;
  }

  LIBC_INLINE static constexpr StorageType encode(Significand value) {
    return value.to_storage_type() & SIG_MASK;
  }

  LIBC_INLINE static constexpr StorageType encode(BiasedExponent exp,
                                                  Significand sig) {
    return encode(exp) | encode(sig);
  }

  LIBC_INLINE static constexpr StorageType encode(bool sign, BiasedExponent exp,
                                                  Significand sig) {
    if (sign)
      return SIGN_MASK | encode(exp, sig);
    return encode(exp, sig);
  }

  LIBC_INLINE constexpr StorageType exp_bits() const { return bits & EXP_MASK; }
  LIBC_INLINE constexpr StorageType sig_bits() const { return bits & SIG_MASK; }
  LIBC_INLINE constexpr StorageType exp_sig_bits() const {
    return bits & EXP_SIG_MASK;
  }

private:
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
  LIBC_INLINE constexpr void set_uintval(StorageType value) {
    bits = (value & FP_MASK);
  }

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

protected:
  using typename UP::BiasedExponent;
  using typename UP::Exponent;
  using typename UP::Significand;
  using UP::encode;
  using UP::exp_bits;
  using UP::exp_sig_bits;
  using UP::sig_bits;

public:
  LIBC_INLINE constexpr bool is_nan() const {
    return exp_sig_bits() >
           encode(BiasedExponent::BITS_ALL_ONES(), Significand::ZERO());
  }
  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return exp_sig_bits() >=
           encode(BiasedExponent::BITS_ALL_ONES(), Significand::MSB());
  }
  LIBC_INLINE constexpr bool is_signaling_nan() const {
    return is_nan() && !is_quiet_nan();
  }
  LIBC_INLINE constexpr bool is_inf() const {
    return exp_sig_bits() ==
           encode(BiasedExponent::BITS_ALL_ONES(), Significand::ZERO());
  }
  LIBC_INLINE constexpr bool is_zero() const {
    return exp_sig_bits() ==
           encode(BiasedExponent::BITS_ALL_ZEROES(), Significand::ZERO());
  }
  LIBC_INLINE constexpr bool is_finite() const {
    return exp_bits() != encode(BiasedExponent::BITS_ALL_ONES());
  }
  LIBC_INLINE
  constexpr bool is_subnormal() const {
    return exp_bits() == encode(BiasedExponent::BITS_ALL_ZEROES());
  }
  LIBC_INLINE constexpr bool is_normal() const {
    return is_finite() && !is_subnormal();
  }

  LIBC_INLINE static constexpr StorageType zero(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ZEROES(), Significand::ZERO());
  }
  LIBC_INLINE static constexpr StorageType one(bool sign = false) {
    return encode(sign, Exponent::ZERO(), Significand::ZERO());
  }
  LIBC_INLINE static constexpr StorageType min_subnormal(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ZEROES(), Significand::LSB());
  }
  LIBC_INLINE static constexpr StorageType max_subnormal(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ZEROES(),
                  Significand::BITS_ALL_ONES());
  }
  LIBC_INLINE static constexpr StorageType min_normal(bool sign = false) {
    return encode(sign, Exponent::MIN(), Significand::ZERO());
  }
  LIBC_INLINE static constexpr StorageType max_normal(bool sign = false) {
    return encode(sign, Exponent::MAX(), Significand::BITS_ALL_ONES());
  }
  LIBC_INLINE static constexpr StorageType inf(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ONES(), Significand::ZERO());
  }
  LIBC_INLINE static constexpr StorageType build_nan(bool sign = false,
                                                     StorageType v = 0) {
    return encode(sign, BiasedExponent::BITS_ALL_ONES(),
                  (v ? Significand(v) : (Significand::MSB() >> 1)));
  }
  LIBC_INLINE static constexpr StorageType build_quiet_nan(bool sign = false,
                                                           StorageType v = 0) {
    return encode(sign, BiasedExponent::BITS_ALL_ONES(),
                  Significand::MSB() | Significand(v));
  }

  // The function return mantissa with the implicit bit set iff the current
  // value is a valid normal number.
  LIBC_INLINE constexpr StorageType get_explicit_mantissa() {
    if (is_subnormal())
      return sig_bits();
    return (StorageType(1) << UP::SIG_LEN) | sig_bits();
  }
};

// Specialization for the X86 Extended Precision type.
template <>
struct FPRep<FPType::X86_Binary80> : public FPRepBase<FPType::X86_Binary80> {
  using UP = FPRepBase<FPType::X86_Binary80>;
  using typename UP::StorageType;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;
  using UP::MANTISSA_PRECISION;

protected:
  using typename UP::BiasedExponent;
  using typename UP::Significand;
  using UP::encode;

public:
  // The x86 80 bit float represents the leading digit of the mantissa
  // explicitly. This is the mask for that bit.
  static constexpr StorageType EXPLICIT_BIT_MASK = StorageType(1)
                                                   << FRACTION_LEN;
  // The X80 significand is made of an explicit bit and the fractional part.
  static_assert((EXPLICIT_BIT_MASK & FRACTION_MASK) == 0,
                "the explicit bit and the fractional part should not overlap");
  static_assert((EXPLICIT_BIT_MASK | FRACTION_MASK) == SIG_MASK,
                "the explicit bit and the fractional part should cover the "
                "whole significand");

  LIBC_INLINE constexpr bool is_nan() const {
    // Most encoding forms from the table found in
    // https://en.wikipedia.org/wiki/Extended_precision#x86_extended_precision_format
    // are interpreted as NaN.
    // More precisely :
    // - Pseudo-Infinity
    // - Pseudo Not a Number
    // - Signalling Not a Number
    // - Floating-point Indefinite
    // - Quiet Not a Number
    // - Unnormal
    // This can be reduced to the following logic:
    if (exp_bits() == encode(BiasedExponent::BITS_ALL_ONES()))
      return !is_inf();
    if (exp_bits() != encode(BiasedExponent::BITS_ALL_ZEROES()))
      return (sig_bits() & encode(Significand::MSB())) == 0;
    return false;
  }
  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return exp_sig_bits() >=
           encode(BiasedExponent::BITS_ALL_ONES(),
                  Significand::MSB() | (Significand::MSB() >> 1));
  }
  LIBC_INLINE constexpr bool is_signaling_nan() const {
    return is_nan() && !is_quiet_nan();
  }
  LIBC_INLINE constexpr bool is_inf() const {
    return exp_sig_bits() ==
           encode(BiasedExponent::BITS_ALL_ONES(), Significand::MSB());
  }
  LIBC_INLINE constexpr bool is_zero() const {
    return exp_sig_bits() ==
           encode(BiasedExponent::BITS_ALL_ZEROES(), Significand::ZERO());
  }
  LIBC_INLINE constexpr bool is_finite() const {
    return !is_inf() && !is_nan();
  }
  LIBC_INLINE
  constexpr bool is_subnormal() const {
    return exp_sig_bits() >
           encode(BiasedExponent::BITS_ALL_ZEROES(), Significand::ZERO());
  }
  LIBC_INLINE constexpr bool is_normal() const {
    const auto exp = exp_bits();
    if (exp == encode(BiasedExponent::BITS_ALL_ZEROES()) ||
        exp == encode(BiasedExponent::BITS_ALL_ONES()))
      return false;
    return get_implicit_bit();
  }

  LIBC_INLINE static constexpr StorageType zero(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ZEROES(), Significand::ZERO());
  }
  LIBC_INLINE static constexpr StorageType one(bool sign = false) {
    return encode(sign, Exponent::ZERO(), Significand::MSB());
  }
  LIBC_INLINE static constexpr StorageType min_subnormal(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ZEROES(), Significand::LSB());
  }
  LIBC_INLINE static constexpr StorageType max_subnormal(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ZEROES(),
                  Significand::BITS_ALL_ONES() ^ Significand::MSB());
  }
  LIBC_INLINE static constexpr StorageType min_normal(bool sign = false) {
    return encode(sign, Exponent::MIN(), Significand::MSB());
  }
  LIBC_INLINE static constexpr StorageType max_normal(bool sign = false) {
    return encode(sign, Exponent::MAX(), Significand::BITS_ALL_ONES());
  }
  LIBC_INLINE static constexpr StorageType inf(bool sign = false) {
    return encode(sign, BiasedExponent::BITS_ALL_ONES(), Significand::MSB());
  }
  LIBC_INLINE static constexpr StorageType build_nan(bool sign = false,
                                                     StorageType v = 0) {
    return encode(sign, BiasedExponent::BITS_ALL_ONES(),
                  Significand::MSB() |
                      (v ? Significand(v) : (Significand::MSB() >> 2)));
  }
  LIBC_INLINE static constexpr StorageType build_quiet_nan(bool sign = false,
                                                           StorageType v = 0) {
    return encode(sign, BiasedExponent::BITS_ALL_ONES(),
                  Significand::MSB() | (Significand::MSB() >> 1) |
                      Significand(v));
  }

  LIBC_INLINE constexpr StorageType get_explicit_mantissa() const {
    return sig_bits();
  }

  // The following functions are specific to FPRep<FPType::X86_Binary80>.
  // TODO: Remove if possible.
  LIBC_INLINE constexpr bool get_implicit_bit() const {
    return static_cast<bool>(bits & EXPLICIT_BIT_MASK);
  }

  LIBC_INLINE constexpr void set_implicit_bit(bool implicitVal) {
    if (get_implicit_bit() != implicitVal)
      bits ^= EXPLICIT_BIT_MASK;
  }
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

// A generic class to represent floating point formats.
// On most platforms, the 'float' type corresponds to single precision
// floating point numbers, the 'double' type corresponds to double precision
// floating point numers, and the 'long double' type corresponds to the quad
// precision floating numbers. On x86 platforms however, the 'long double'
// type maps to an x87 floating point format.
template <typename T> struct FPBits : public internal::FPRep<get_fp_type<T>()> {
  static_assert(cpp::is_floating_point_v<T>,
                "FPBits instantiated with invalid type.");
  using UP = internal::FPRep<get_fp_type<T>()>;
  using Rep = UP;
  using StorageType = typename UP::StorageType;

  using UP::bits;
  using UP::EXP_LEN;
  using UP::UP;

  // Constants.
  static constexpr int MAX_BIASED_EXPONENT = (1 << EXP_LEN) - 1;
  static constexpr StorageType MIN_NORMAL = UP::min_normal(false);
  static constexpr StorageType MAX_NORMAL = UP::max_normal(false);
  static constexpr StorageType MIN_SUBNORMAL = UP::min_subnormal(false);
  static constexpr StorageType MAX_SUBNORMAL = UP::max_subnormal(false);

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

  LIBC_INLINE constexpr bool is_inf_or_nan() const { return !UP::is_finite(); }

  LIBC_INLINE constexpr FPBits abs() const {
    return FPBits(bits & UP::EXP_SIG_MASK);
  }

  // Methods below this are used by tests.

  LIBC_INLINE static constexpr T zero(bool sign = false) {
    return FPBits(UP::zero(sign)).get_val();
  }

  LIBC_INLINE static constexpr T neg_zero() { return zero(true); }

  LIBC_INLINE static constexpr T inf(bool sign = false) {
    return FPBits(UP::inf(sign)).get_val();
  }

  LIBC_INLINE static constexpr T neg_inf() { return inf(true); }

  LIBC_INLINE static constexpr T min_normal() {
    return FPBits(UP::min_normal(false)).get_val();
  }

  LIBC_INLINE static constexpr T max_normal() {
    return FPBits(UP::max_normal(false)).get_val();
  }

  LIBC_INLINE static constexpr T min_denormal() {
    return FPBits(UP::min_subnormal(false)).get_val();
  }

  LIBC_INLINE static constexpr T max_denormal() {
    return FPBits(UP::max_subnormal(false)).get_val();
  }

  LIBC_INLINE static constexpr T build_nan(StorageType v) {
    return FPBits(UP::build_nan(false, v)).get_val();
  }

  LIBC_INLINE static constexpr T build_quiet_nan(StorageType v) {
    return FPBits(UP::build_quiet_nan(false, v)).get_val();
  }

  // TODO: Use an uint32_t for 'biased_exp'.
  LIBC_INLINE static constexpr FPBits<T>
  create_value(bool sign, StorageType biased_exp, StorageType mantissa) {
    static_assert(get_fp_type<T>() != FPType::X86_Binary80,
                  "This function is not tested for X86 Extended Precision");
    return FPBits(UP::encode(
        sign, typename UP::BiasedExponent(static_cast<uint32_t>(biased_exp)),
        typename UP::Significand(mantissa)));
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
    static_assert(get_fp_type<T>() != FPType::X86_Binary80,
                  "This function is not tested for X86 Extended Precision");
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

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FPBITS_H
