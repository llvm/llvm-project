//===-- Definition of float128 data type  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/comparison_operations.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/FPUtil/generic/div.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/UInt128.h"

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct Float128 {
  UInt128 bits;

  LIBC_INLINE Float128() = default;

  template <typename T>
  LIBC_INLINE constexpr explicit Float128(T value)
      : bits(static_cast<UInt128>(0U)) {
    if constexpr (cpp::is_floating_point_v<T>) {
      bits = fputil::cast<Float128>(value).bits;
    } else if constexpr (cpp::is_integral_v<T>) {
      Sign sign = Sign::POS;

      if constexpr (cpp::is_signed_v<T>) {
        if (value < 0) {
          sign = Sign::NEG;
          value = -value;
        }
      }

      fputil::DyadicFloat<cpp::numeric_limits<cpp::make_unsigned_t<T>>::digits>
          xd(sign, 0, value);
      bits = xd.template as<Float128, /*ShouldSignalExceptions=*/true>().bits;

    } else if constexpr (cpp::is_convertible_v<T, Float128>) {
      bits = value.operator Float128().bits;
    } else {
      bits = fputil::cast<Float128>(static_cast<float>(value)).bits;
    }
  }

  template <cpp::enable_if_t<fputil::get_fp_type<float>() ==
                                 fputil::FPType::IEEE754_Binary32,
                             int> = 0>
  LIBC_INLINE constexpr operator float() const {
    UInt128 x_bits = static_cast<UInt128>(bits) << 16U;
    return cpp::bit_cast<float>(x_bits);
  }

  template <typename T, cpp::enable_if_t<cpp::is_integral_v<T>, int> = 0>
  LIBC_INLINE constexpr explicit operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }

  LIBC_INLINE constexpr bool operator==(BFloat16 other) const {
    return fputil::equals(*this, other);
  }

  LIBC_INLINE constexpr bool operator!=(BFloat16 other) const {
    return !fputil::equals(*this, other);
  }

  LIBC_INLINE constexpr bool operator<(BFloat16 other) const {
    return fputil::less_than(*this, other);
  }

  LIBC_INLINE constexpr bool operator<=(BFloat16 other) const {
    return fputil::less_than_or_equals(*this, other);
  }

  LIBC_INLINE constexpr bool operator>(BFloat16 other) const {
    return fputil::greater_than(*this, other);
  }

  LIBC_INLINE constexpr bool operator>=(BFloat16 other) const {
    return fputil::greater_than_or_equals(*this, other);
  }

  LIBC_INLINE LIBC_BIT_CAST_CONSTEXPR BFloat16 operator-() const {
    fputil::FPBits<bfloat16> result(*this);
    result.set_sign(result.is_pos() ? Sign::NEG : Sign::POS);
    return result.get_val();
  }

  LIBC_INLINE constexpr BFloat16 operator+(BFloat16 other) const {
    return fputil::generic::add<BFloat16>(*this, other);
  }

  LIBC_INLINE constexpr BFloat16 operator-(BFloat16 other) const {
    return fputil::generic::sub<BFloat16>(*this, other);
  }

  LIBC_INLINE constexpr BFloat16 operator*(BFloat16 other) const {
    return fputil::generic::mul<bfloat16>(*this, other);
  }

  LIBC_INLINE constexpr BFloat16 operator/(BFloat16 other) const {
    return fputil::generic::div<bfloat16>(*this, other);
  }

  LIBC_INLINE constexpr BFloat16 &operator*=(const BFloat16 &other) {
    *this = *this * other;
    return *this;
  }
}; // struct BFloat16

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
