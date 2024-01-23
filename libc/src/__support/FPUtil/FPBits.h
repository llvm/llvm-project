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

// A type to interact with floating point type signs.
// This may be moved outside of 'fputil' if useful.
struct Sign {
  LIBC_INLINE constexpr bool is_pos() const { return !is_negative; }
  LIBC_INLINE constexpr bool is_neg() const { return is_negative; }

  LIBC_INLINE friend constexpr bool operator==(Sign a, Sign b) {
    return a.is_negative == b.is_negative;
  }
  LIBC_INLINE friend constexpr bool operator!=(Sign a, Sign b) {
    return !(a == b);
  }

  static const Sign POS;
  static const Sign NEG;

private:
  LIBC_INLINE constexpr explicit Sign(bool is_negative)
      : is_negative(is_negative) {}

  bool is_negative;
};

LIBC_INLINE_VAR constexpr Sign Sign::NEG = Sign(true);
LIBC_INLINE_VAR constexpr Sign Sign::POS = Sign(false);

// The classes hierarchy is as follows:
//
//             ┌───────────────────┐
//             │ FPLayout<FPType>  │
//             └─────────▲─────────┘
//                       │
//             ┌─────────┴─────────┐
//             │ FPStorage<FPType> │
//             └─────────▲─────────┘
//                       │
//          ┌────────────┴─────────────┐
//          │                          │
// ┌────────┴─────────┐ ┌──────────────┴──────────────────┐
// │ FPRepSem<FPType> │ │  FPRepSem<FPType::X86_Binary80  │
// └────────▲─────────┘ └──────────────▲──────────────────┘
//          │                          │
//          └────────────┬─────────────┘
//                       │
//                 ┌─────┴─────┐
//                 │  FPRep<T> │
//                 └───────────┘
//                       │
//                 ┌─────┴─────┐
//                 │ FPBits<T> │
//                 └───────────┘
//
// - 'FPLayout' defines only a few constants, namely the 'StorageType' and
// length of the sign, the exponent, fraction and significand parts.
// - 'FPStorage' builds more constants on top of those from 'FPLayout' like
// exponent bias and masks. It also holds the bit representation of the
// floating point as a 'StorageType' type and defines tools to assemble or test
// these parts.
// - 'FPRepSem' defines functions to interact semantically with the floating
// point representation. The default implementation is the one for 'IEEE754', a
// specialization is provided for X86 Extended Precision.
// - 'FPRep' derives from 'FPRepSem' and adds functions that are common to all
// implementations.
// - 'FPBits' exposes all functions from 'FPRep' but operates on the native C++
// floating point type instead of 'FPType'.

namespace internal {

// Defines the layout (sign, exponent, significand) of a floating point type in
// memory. It also defines its associated StorageType, i.e., the unsigned
// integer type used to manipulate its representation.
// Additionally we provide the fractional part length, i.e., the number of bits
// after the decimal dot when the number is in normal form.
template <FPType> struct FPLayout {};

template <> struct FPLayout<FPType::IEEE754_Binary16> {
  using StorageType = uint16_t;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 5;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 10;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SIG_LEN;
};

template <> struct FPLayout<FPType::IEEE754_Binary32> {
  using StorageType = uint32_t;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 8;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 23;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SIG_LEN;
};

template <> struct FPLayout<FPType::IEEE754_Binary64> {
  using StorageType = uint64_t;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 11;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 52;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SIG_LEN;
};

template <> struct FPLayout<FPType::IEEE754_Binary128> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 112;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SIG_LEN;
};

template <> struct FPLayout<FPType::X86_Binary80> {
  using StorageType = UInt128;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int EXP_LEN = 15;
  LIBC_INLINE_VAR static constexpr int SIG_LEN = 64;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SIG_LEN - 1;
};

// FPStorage derives useful constants from the FPLayout above.
template <FPType fp_type> struct FPStorage : public FPLayout<fp_type> {
  using UP = FPLayout<fp_type>;

  using UP::EXP_LEN;  // The number of bits for the *exponent* part
  using UP::SIG_LEN;  // The number of bits for the *significand* part
  using UP::SIGN_LEN; // The number of bits for the *sign* part
  // For convenience, the sum of `SIG_LEN`, `EXP_LEN`, and `SIGN_LEN`.
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN = SIGN_LEN + EXP_LEN + SIG_LEN;

  // The number of bits after the decimal dot when the number is in normal form.
  using UP::FRACTION_LEN;

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

  // The bit pattern that keeps only the *significand* part.
  LIBC_INLINE_VAR static constexpr StorageType SIG_MASK =
      mask_trailing_ones<StorageType, SIG_LEN>();
  // The bit pattern that keeps only the *exponent* part.
  LIBC_INLINE_VAR static constexpr StorageType EXP_MASK =
      mask_trailing_ones<StorageType, EXP_LEN>() << SIG_LEN;
  // The bit pattern that keeps only the *sign* part.
  LIBC_INLINE_VAR static constexpr StorageType SIGN_MASK =
      mask_trailing_ones<StorageType, SIGN_LEN>() << (EXP_LEN + SIG_LEN);
  // The bit pattern that keeps only the *exponent + significand* part.
  LIBC_INLINE_VAR static constexpr StorageType EXP_SIG_MASK =
      mask_trailing_ones<StorageType, EXP_LEN + SIG_LEN>();
  // The bit pattern that keeps only the *sign + exponent + significand* part.
  LIBC_INLINE_VAR static constexpr StorageType FP_MASK =
      mask_trailing_ones<StorageType, TOTAL_LEN>();
  // The bit pattern that keeps only the *fraction* part.
  // i.e., the *significand* without the leading one.
  LIBC_INLINE_VAR static constexpr StorageType FRACTION_MASK =
      mask_trailing_ones<StorageType, FRACTION_LEN>();

  static_assert((SIG_MASK & EXP_MASK & SIGN_MASK) == 0, "masks disjoint");
  static_assert((SIG_MASK | EXP_MASK | SIGN_MASK) == FP_MASK, "masks cover");

protected:
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
  // as they are in the range [ZERO, BITS_ALL_ONES].
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
      return Significand(StorageType(1) << (SIG_LEN - 1));
    }
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

  LIBC_INLINE static constexpr StorageType encode(Sign sign, BiasedExponent exp,
                                                  Significand sig) {
    if (sign.is_neg())
      return SIGN_MASK | encode(exp, sig);
    return encode(exp, sig);
  }

  // The floating point number representation as an unsigned integer.
  StorageType bits{};

  LIBC_INLINE constexpr FPStorage() : bits(0) {}
  LIBC_INLINE constexpr FPStorage(StorageType value) : bits(value) {}

  // Observers
  LIBC_INLINE constexpr StorageType exp_bits() const { return bits & EXP_MASK; }
  LIBC_INLINE constexpr StorageType sig_bits() const { return bits & SIG_MASK; }
  LIBC_INLINE constexpr StorageType exp_sig_bits() const {
    return bits & EXP_SIG_MASK;
  }
};

// This layer defines all functions that are specific to how the the floating
// point type is encoded. It enables constructions, modification and observation
// of values manipulated as 'StorageType'.
template <FPType fp_type, typename RetT>
struct FPRepSem : public FPStorage<fp_type> {
  using UP = FPStorage<fp_type>;
  using typename UP::StorageType;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;

protected:
  using BiasedExp = typename UP::BiasedExponent;
  using Exp = typename UP::Exponent;
  using Sig = typename UP::Significand;
  using UP::encode;
  using UP::exp_bits;
  using UP::exp_sig_bits;
  using UP::sig_bits;
  using UP::UP;

public:
  // Builders
  LIBC_INLINE static constexpr RetT one(Sign sign = Sign::POS) {
    return RetT(encode(sign, Exp::ZERO(), Sig::ZERO()));
  }
  LIBC_INLINE static constexpr RetT min_subnormal(Sign sign = Sign::POS) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ZEROES(), Sig::LSB()));
  }
  LIBC_INLINE static constexpr RetT max_subnormal(Sign sign = Sign::POS) {
    return RetT(
        encode(sign, BiasedExp::BITS_ALL_ZEROES(), Sig::BITS_ALL_ONES()));
  }
  LIBC_INLINE static constexpr RetT min_normal(Sign sign = Sign::POS) {
    return RetT(encode(sign, Exp::MIN(), Sig::ZERO()));
  }
  LIBC_INLINE static constexpr RetT max_normal(Sign sign = Sign::POS) {
    return RetT(encode(sign, Exp::MAX(), Sig::BITS_ALL_ONES()));
  }
  LIBC_INLINE static constexpr RetT inf(Sign sign = Sign::POS) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ONES(), Sig::ZERO()));
  }
  LIBC_INLINE static constexpr RetT build_nan(Sign sign = Sign::POS,
                                              StorageType v = 0) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ONES(),
                       (v ? Sig(v) : (Sig::MSB() >> 1))));
  }
  LIBC_INLINE static constexpr RetT build_quiet_nan(Sign sign = Sign::POS,
                                                    StorageType v = 0) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ONES(), Sig::MSB() | Sig(v)));
  }

  // Observers
  LIBC_INLINE constexpr bool is_nan() const {
    return exp_sig_bits() > encode(BiasedExp::BITS_ALL_ONES(), Sig::ZERO());
  }
  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return exp_sig_bits() >= encode(BiasedExp::BITS_ALL_ONES(), Sig::MSB());
  }
  LIBC_INLINE constexpr bool is_signaling_nan() const {
    return is_nan() && !is_quiet_nan();
  }
  LIBC_INLINE constexpr bool is_inf() const {
    return exp_sig_bits() == encode(BiasedExp::BITS_ALL_ONES(), Sig::ZERO());
  }
  LIBC_INLINE constexpr bool is_finite() const {
    return exp_bits() != encode(BiasedExp::BITS_ALL_ONES());
  }
  LIBC_INLINE
  constexpr bool is_subnormal() const {
    return exp_bits() == encode(BiasedExp::BITS_ALL_ZEROES());
  }
  LIBC_INLINE constexpr bool is_normal() const {
    return is_finite() && !is_subnormal();
  }
  // Returns the mantissa with the implicit bit set iff the current
  // value is a valid normal number.
  LIBC_INLINE constexpr StorageType get_explicit_mantissa() {
    if (is_subnormal())
      return sig_bits();
    return (StorageType(1) << UP::SIG_LEN) | sig_bits();
  }
};

// Specialization for the X86 Extended Precision type.
template <typename RetT>
struct FPRepSem<FPType::X86_Binary80, RetT>
    : public FPStorage<FPType::X86_Binary80> {
  using UP = FPStorage<FPType::X86_Binary80>;
  using typename UP::StorageType;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;

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

protected:
  using BiasedExp = typename UP::BiasedExponent;
  using Sig = typename UP::Significand;
  using UP::encode;
  using UP::UP;

public:
  // Builders
  LIBC_INLINE static constexpr RetT one(Sign sign = Sign::POS) {
    return RetT(encode(sign, Exponent::ZERO(), Sig::MSB()));
  }
  LIBC_INLINE static constexpr RetT min_subnormal(Sign sign = Sign::POS) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ZEROES(), Sig::LSB()));
  }
  LIBC_INLINE static constexpr RetT max_subnormal(Sign sign = Sign::POS) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ZEROES(),
                       Sig::BITS_ALL_ONES() ^ Sig::MSB()));
  }
  LIBC_INLINE static constexpr RetT min_normal(Sign sign = Sign::POS) {
    return RetT(encode(sign, Exponent::MIN(), Sig::MSB()));
  }
  LIBC_INLINE static constexpr RetT max_normal(Sign sign = Sign::POS) {
    return RetT(encode(sign, Exponent::MAX(), Sig::BITS_ALL_ONES()));
  }
  LIBC_INLINE static constexpr RetT inf(Sign sign = Sign::POS) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ONES(), Sig::MSB()));
  }
  LIBC_INLINE static constexpr RetT build_nan(Sign sign = Sign::POS,
                                              StorageType v = 0) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ONES(),
                       Sig::MSB() | (v ? Sig(v) : (Sig::MSB() >> 2))));
  }
  LIBC_INLINE static constexpr RetT build_quiet_nan(Sign sign = Sign::POS,
                                                    StorageType v = 0) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ONES(),
                       Sig::MSB() | (Sig::MSB() >> 1) | Sig(v)));
  }

  // Observers
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
    if (exp_bits() == encode(BiasedExp::BITS_ALL_ONES()))
      return !is_inf();
    if (exp_bits() != encode(BiasedExp::BITS_ALL_ZEROES()))
      return (sig_bits() & encode(Sig::MSB())) == 0;
    return false;
  }
  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return exp_sig_bits() >=
           encode(BiasedExp::BITS_ALL_ONES(), Sig::MSB() | (Sig::MSB() >> 1));
  }
  LIBC_INLINE constexpr bool is_signaling_nan() const {
    return is_nan() && !is_quiet_nan();
  }
  LIBC_INLINE constexpr bool is_inf() const {
    return exp_sig_bits() == encode(BiasedExp::BITS_ALL_ONES(), Sig::MSB());
  }
  LIBC_INLINE constexpr bool is_finite() const {
    return !is_inf() && !is_nan();
  }
  LIBC_INLINE
  constexpr bool is_subnormal() const {
    return exp_bits() == encode(BiasedExp::BITS_ALL_ZEROES());
  }
  LIBC_INLINE constexpr bool is_normal() const {
    const auto exp = exp_bits();
    if (exp == encode(BiasedExp::BITS_ALL_ZEROES()) ||
        exp == encode(BiasedExp::BITS_ALL_ONES()))
      return false;
    return get_implicit_bit();
  }
  LIBC_INLINE constexpr StorageType get_explicit_mantissa() const {
    return sig_bits();
  }

  // This functions is specific to FPRepSem<FPType::X86_Binary80>.
  // TODO: Remove if possible.
  LIBC_INLINE constexpr bool get_implicit_bit() const {
    return static_cast<bool>(bits & EXPLICIT_BIT_MASK);
  }

  // This functions is specific to FPRepSem<FPType::X86_Binary80>.
  // TODO: Remove if possible.
  LIBC_INLINE constexpr void set_implicit_bit(bool implicitVal) {
    if (get_implicit_bit() != implicitVal)
      bits ^= EXPLICIT_BIT_MASK;
  }
};

// 'FPRep' is the bottom of the class hierarchy that only deals with 'FPType'.
// The operations dealing with specific float semantics are implemented by
// 'FPRepSem' above and specialized when needed.
//
// The 'RetT' type is being propagated up to 'FPRepSem' so that the functions
// creating new values (Builders) can return the appropriate type. That is, when
// creating a value through 'FPBits' below the builder will return an 'FPBits'
// value:
// i.e., FPBits<float>::zero() // returns an FPBits<float>
// When we don't care about specific C++ floating point type we can use 'FPRep'
// directly and 'RetT' defaults to 'StorageType':
// i.e., FPRep<FPType:IEEE754_Binary32:>::zero() // returns an 'uint32_t'
template <FPType fp_type,
          typename RetT = typename FPLayout<fp_type>::StorageType>
struct FPRep : public FPRepSem<fp_type, RetT> {
  using UP = FPRepSem<fp_type, RetT>;
  using StorageType = typename UP::StorageType;

protected:
  using UP::bits;
  using UP::encode;
  using UP::exp_bits;
  using UP::exp_sig_bits;

  using BiasedExp = typename UP::BiasedExponent;
  using Sig = typename UP::Significand;

  using UP::FP_MASK;
  using UP::SIG_LEN;

public:
  using UP::EXP_BIAS;
  using UP::EXP_MASK;
  using UP::FRACTION_MASK;
  using UP::SIGN_MASK;

  // Comparison
  LIBC_INLINE constexpr friend bool operator==(FPRep a, FPRep b) {
    return a.uintval() == b.uintval();
  }
  LIBC_INLINE constexpr friend bool operator!=(FPRep a, FPRep b) {
    return a.uintval() != b.uintval();
  }

  // Representation
  LIBC_INLINE constexpr StorageType uintval() const { return bits & FP_MASK; }
  LIBC_INLINE constexpr void set_uintval(StorageType value) {
    bits = (value & FP_MASK);
  }

  // Builders
  LIBC_INLINE static constexpr RetT zero(Sign sign = Sign::POS) {
    return RetT(encode(sign, BiasedExp::BITS_ALL_ZEROES(), Sig::ZERO()));
  }
  using UP::build_nan;
  using UP::build_quiet_nan;
  using UP::inf;
  using UP::max_normal;
  using UP::max_subnormal;
  using UP::min_normal;
  using UP::min_subnormal;
  using UP::one;

  // Modifiers
  LIBC_INLINE constexpr RetT abs() const {
    return RetT(bits & UP::EXP_SIG_MASK);
  }

  // Observers
  using UP::get_explicit_mantissa;
  LIBC_INLINE constexpr bool is_zero() const { return exp_sig_bits() == 0; }
  LIBC_INLINE constexpr bool is_inf_or_nan() const { return !is_finite(); }
  using UP::is_finite;
  using UP::is_inf;
  using UP::is_nan;
  using UP::is_normal;
  using UP::is_quiet_nan;
  using UP::is_signaling_nan;
  using UP::is_subnormal;
  LIBC_INLINE constexpr bool is_neg() const { return sign().is_neg(); }
  LIBC_INLINE constexpr bool is_pos() const { return sign().is_pos(); }

  // Parts
  LIBC_INLINE constexpr Sign sign() const {
    return (bits & SIGN_MASK) ? Sign::NEG : Sign::POS;
  }

  LIBC_INLINE constexpr void set_sign(Sign signVal) {
    if (sign() != signVal)
      bits ^= SIGN_MASK;
  }

  LIBC_INLINE constexpr uint16_t get_biased_exponent() const {
    return uint16_t((bits & UP::EXP_MASK) >> UP::SIG_LEN);
  }

  LIBC_INLINE constexpr void set_biased_exponent(StorageType biased) {
    bits = merge(bits, biased << SIG_LEN, EXP_MASK);
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

  LIBC_INLINE constexpr StorageType get_mantissa() const {
    return bits & FRACTION_MASK;
  }

  LIBC_INLINE constexpr void set_mantissa(StorageType mantVal) {
    bits = merge(bits, mantVal, FRACTION_MASK);
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

// A generic class to manipulate floating point formats.
// It derives most of its functionality to FPRep above.
template <typename T>
struct FPBits final : public internal::FPRep<get_fp_type<T>(), FPBits<T>> {
  static_assert(cpp::is_floating_point_v<T>,
                "FPBits instantiated with invalid type.");
  using UP = internal::FPRep<get_fp_type<T>(), FPBits<T>>;
  using Rep = UP;
  using StorageType = typename UP::StorageType;

  using UP::bits;

  // Constants.
  LIBC_INLINE_VAR static constexpr int MAX_BIASED_EXPONENT =
      (1 << UP::EXP_LEN) - 1;

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

  // TODO: Use an uint32_t for 'biased_exp'.
  LIBC_INLINE static constexpr FPBits<T>
  create_value(Sign sign, StorageType biased_exp, StorageType mantissa) {
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
    int lz = cpp::countl_zero(number) - UP::EXP_LEN;
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
