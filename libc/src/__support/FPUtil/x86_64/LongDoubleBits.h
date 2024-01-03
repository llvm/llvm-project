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

template <>
struct FPBits<long double> : public internal::FPRep<FPType::X86_Binary80> {
  using UP = internal::FPRep<FPType::X86_Binary80>;
  using StorageType = typename UP::StorageType;

private:
  using UP::bits;
  using UP::EXP_SIG_MASK;
  using UP::QUIET_NAN_MASK;

public:
  // Constants.
  static constexpr int MAX_BIASED_EXPONENT = (1 << EXP_LEN) - 1;
  // The x86 80 bit float represents the leading digit of the mantissa
  // explicitly. This is the mask for that bit.
  static constexpr StorageType EXPLICIT_BIT_MASK = StorageType(1)
                                                   << FRACTION_LEN;
  static_assert((EXPLICIT_BIT_MASK & FRACTION_MASK) == 0, "mask disjoint");
  static_assert((EXPLICIT_BIT_MASK | FRACTION_MASK) == SIG_MASK, "mask cover");
  static constexpr StorageType MIN_SUBNORMAL = StorageType(1);
  static constexpr StorageType MAX_SUBNORMAL = FRACTION_MASK;
  static constexpr StorageType MIN_NORMAL =
      (StorageType(1) << SIG_LEN) | EXPLICIT_BIT_MASK;
  static constexpr StorageType MAX_NORMAL =
      (StorageType(MAX_BIASED_EXPONENT - 1) << SIG_LEN) | SIG_MASK;

  // Constructors.
  LIBC_INLINE constexpr FPBits() = default;

  template <typename XType> LIBC_INLINE constexpr explicit FPBits(XType x) {
    using Unqual = typename cpp::remove_cv_t<XType>;
    if constexpr (cpp::is_same_v<Unqual, long double>) {
      bits = cpp::bit_cast<StorageType>(x);
    } else if constexpr (cpp::is_same_v<Unqual, StorageType>) {
      bits = x;
    } else {
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

  LIBC_INLINE constexpr StorageType get_explicit_mantissa() const {
    return bits & SIG_MASK;
  }

  LIBC_INLINE constexpr bool get_implicit_bit() const {
    return bits & EXPLICIT_BIT_MASK;
  }

  LIBC_INLINE constexpr void set_implicit_bit(bool implicitVal) {
    if (get_implicit_bit() != implicitVal)
      bits ^= EXPLICIT_BIT_MASK;
  }

  LIBC_INLINE constexpr bool is_inf() const {
    return get_biased_exponent() == MAX_BIASED_EXPONENT &&
           get_mantissa() == 0 && get_implicit_bit() == 1;
  }

  LIBC_INLINE constexpr bool is_nan() const {
    if (get_biased_exponent() == MAX_BIASED_EXPONENT) {
      return (get_implicit_bit() == 0) || get_mantissa() != 0;
    } else if (get_biased_exponent() != 0) {
      return get_implicit_bit() == 0;
    }
    return false;
  }

  LIBC_INLINE constexpr bool is_inf_or_nan() const {
    return (get_biased_exponent() == MAX_BIASED_EXPONENT) ||
           (get_biased_exponent() != 0 && get_implicit_bit() == 0);
  }

  // Methods below this are used by tests.

  LIBC_INLINE static constexpr long double zero() { return 0.0l; }

  LIBC_INLINE static constexpr long double neg_zero() { return -0.0l; }

  LIBC_INLINE static constexpr long double inf(bool sign = false) {
    FPBits<long double> bits(0.0l);
    bits.set_biased_exponent(MAX_BIASED_EXPONENT);
    bits.set_implicit_bit(1);
    if (sign) {
      bits.set_sign(true);
    }
    return bits.get_val();
  }

  LIBC_INLINE static constexpr long double neg_inf() { return inf(true); }

  LIBC_INLINE static constexpr long double build_nan(StorageType v) {
    FPBits<long double> bits(0.0l);
    bits.set_biased_exponent(MAX_BIASED_EXPONENT);
    bits.set_implicit_bit(1);
    bits.set_mantissa(v);
    return bits;
  }

  LIBC_INLINE static constexpr long double build_quiet_nan(StorageType v) {
    return build_nan(QUIET_NAN_MASK | v);
  }

  LIBC_INLINE static constexpr long double min_normal() {
    return FPBits(MIN_NORMAL).get_val();
  }

  LIBC_INLINE static constexpr long double max_normal() {
    return FPBits(MAX_NORMAL).get_val();
  }

  LIBC_INLINE static constexpr long double min_denormal() {
    return FPBits(MIN_SUBNORMAL).get_val();
  }

  LIBC_INLINE static constexpr long double max_denormal() {
    return FPBits(MAX_SUBNORMAL).get_val();
  }

  LIBC_INLINE static constexpr FPBits<long double>
  create_value(bool sign, StorageType biased_exp, StorageType mantissa) {
    FPBits<long double> result;
    result.set_sign(sign);
    result.set_biased_exponent(biased_exp);
    result.set_mantissa(mantissa);
    return result;
  }
};

static_assert(
    sizeof(FPBits<long double>) == sizeof(long double),
    "Internal long double representation does not match the machine format.");

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_X86_64_LONGDOUBLEBITS_H
