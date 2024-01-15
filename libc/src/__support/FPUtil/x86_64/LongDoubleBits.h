//===-- Bit representation of x86 long double numbers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_X86_64_LONGDOUBLEBITS_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_X86_64_LONGDOUBLEBITS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/UInt128.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/properties/architectures.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid include"
#endif

#include "src/__support/FPUtil/FPBits.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace fputil {

namespace internal {

template <>
struct FPRep<FPType::X86_Binary80> : public FPRepBase<FPType::X86_Binary80> {
  using UP = FPRepBase<FPType::X86_Binary80>;
  using typename UP::StorageType;
  using UP::FRACTION_LEN;
  using UP::FRACTION_MASK;
  using UP::MANTISSA_PRECISION;

protected:
  using typename UP::Exp;
  using typename UP::Sig;
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
    if (exp_bits() == encode(Exp::BITS_ALL_ONES))
      return !is_inf();
    if (exp_bits() != encode(Exp::BITS_ALL_ZEROES))
      return (sig_bits() & encode(Sig::MSB)) == 0;
    return false;
  }
  LIBC_INLINE constexpr bool is_quiet_nan() const {
    return exp_sig_bits() >=
           encode(Exp::BITS_ALL_ONES, Sig::MSB | (Sig::MSB >> 1));
  }
  LIBC_INLINE constexpr bool is_signaling_nan() const {
    return is_nan() && !is_quiet_nan();
  }
  LIBC_INLINE constexpr bool is_inf() const {
    return exp_sig_bits() == encode(Exp::BITS_ALL_ONES, Sig::MSB);
  }
  LIBC_INLINE constexpr bool is_zero() const {
    return exp_sig_bits() == encode(Exp::BITS_ALL_ZEROES, Sig::BITS_ALL_ZEROES);
  }
  LIBC_INLINE constexpr bool is_finite() const {
    return !is_inf() && !is_nan();
  }
  LIBC_INLINE
  constexpr bool is_subnormal() const {
    return exp_sig_bits() > encode(Exp::BITS_ALL_ZEROES, Sig::BITS_ALL_ZEROES);
  }
  LIBC_INLINE constexpr bool is_normal() const {
    const auto exp = exp_bits();
    if (exp == encode(Exp::BITS_ALL_ZEROES) ||
        exp == encode(Exp::BITS_ALL_ONES))
      return false;
    return get_implicit_bit();
  }

  LIBC_INLINE static constexpr StorageType zero(bool sign = false) {
    return encode(sign, Exp::BITS_ALL_ZEROES, Sig::BITS_ALL_ZEROES);
  }
  LIBC_INLINE static constexpr StorageType one(bool sign = false) {
    return encode(sign, Exp::ZERO, Sig::MSB);
  }
  LIBC_INLINE static constexpr StorageType min_subnormal(bool sign = false) {
    return encode(sign, Exp::BITS_ALL_ZEROES, Sig::ONE);
  }
  LIBC_INLINE static constexpr StorageType max_subnormal(bool sign = false) {
    return encode(sign, Exp::BITS_ALL_ZEROES, Sig::BITS_ALL_ONES ^ Sig::MSB);
  }
  LIBC_INLINE static constexpr StorageType min_normal(bool sign = false) {
    return encode(sign, Exp::MIN, Sig::MSB);
  }
  LIBC_INLINE static constexpr StorageType max_normal(bool sign = false) {
    return encode(sign, Exp::MAX, Sig::BITS_ALL_ONES);
  }
  LIBC_INLINE static constexpr StorageType inf(bool sign = false) {
    return encode(sign, Exp::BITS_ALL_ONES, Sig::MSB);
  }
  LIBC_INLINE static constexpr StorageType build_nan(bool sign = false,
                                                     StorageType v = 0) {
    return encode(sign, Exp::BITS_ALL_ONES,
                  Sig::MSB | (v ? Sig{v} : (Sig::MSB >> 2)));
  }
  LIBC_INLINE static constexpr StorageType build_quiet_nan(bool sign = false,
                                                           StorageType v = 0) {
    return encode(sign, Exp::BITS_ALL_ONES,
                  Sig::MSB | (Sig::MSB >> 1) | Sig{v});
  }

  LIBC_INLINE constexpr StorageType get_explicit_mantissa() const {
    return sig_bits();
  }

  LIBC_INLINE constexpr bool get_implicit_bit() const {
    return bits & EXPLICIT_BIT_MASK;
  }

  LIBC_INLINE constexpr void set_implicit_bit(bool implicitVal) {
    if (get_implicit_bit() != implicitVal)
      bits ^= EXPLICIT_BIT_MASK;
  }
};
} // namespace internal

template <>
struct FPBits<long double> : public internal::FPRep<FPType::X86_Binary80> {
  using UP = internal::FPRep<FPType::X86_Binary80>;
  using Rep = UP;
  using StorageType = typename UP::StorageType;

private:
  using UP::bits;
  using UP::EXP_SIG_MASK;

public:
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
    if constexpr (cpp::is_same_v<Unqual, long double>) {
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
  LIBC_INLINE constexpr long double get_val() const {
    return cpp::bit_cast<long double>(bits);
  }

  LIBC_INLINE constexpr operator long double() const {
    return cpp::bit_cast<long double>(bits);
  }

  LIBC_INLINE constexpr bool is_inf_or_nan() const { return !UP::is_finite(); }

  // Methods below this are used by tests.

  LIBC_INLINE static constexpr long double zero(bool sign = false) {
    return FPBits(UP::zero(sign)).get_val();
  }

  LIBC_INLINE static constexpr long double neg_zero() { return zero(true); }

  LIBC_INLINE static constexpr long double inf(bool sign = false) {
    return FPBits(UP::inf(sign)).get_val();
  }

  LIBC_INLINE static constexpr long double neg_inf() { return inf(true); }

  LIBC_INLINE static constexpr long double min_normal() {
    return FPBits(UP::min_normal(false)).get_val();
  }

  LIBC_INLINE static constexpr long double max_normal() {
    return FPBits(UP::max_normal(false)).get_val();
  }

  LIBC_INLINE static constexpr long double min_denormal() {
    return FPBits(UP::min_subnormal(false)).get_val();
  }

  LIBC_INLINE static constexpr long double max_denormal() {
    return FPBits(UP::max_subnormal(false)).get_val();
  }

  LIBC_INLINE static constexpr long double build_nan(StorageType v) {
    return FPBits(UP::build_nan(false, v)).get_val();
  }

  LIBC_INLINE static constexpr long double build_quiet_nan(StorageType v) {
    return FPBits(UP::build_quiet_nan(false, v)).get_val();
  }
};

static_assert(
    sizeof(FPBits<long double>) == sizeof(long double),
    "Internal long double representation does not match the machine format.");

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_X86_64_LONGDOUBLEBITS_H
