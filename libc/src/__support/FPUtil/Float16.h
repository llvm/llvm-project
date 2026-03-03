//===-- Definition of float16 data type. ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT16_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT16_H

#include "src/__support/macros/properties/types.h"

#ifdef LIBC_USE_SOFT_FLOAT16

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/comparison_operations.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/FPUtil/generic/div.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct Float16 {
  uint16_t bits;

  LIBC_INLINE Float16() = default;

  template <typename T>
  LIBC_INLINE constexpr explicit Float16(T value)
      : bits(static_cast<uint16_t>(0U)) {
    if constexpr (cpp::is_floating_point_v<T>) {
      bits = fputil::cast<Float16>(value).bits;
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
      bits = xd.template as<Float16, /*ShouldSignalExceptions=*/true>().bits;

    } else if constexpr (cpp::is_convertible_v<T, Float16>) {
      bits = value.operator Float16().bits;
    }
  }

  template <cpp::enable_if_t<fputil::get_fp_type<float>() ==
                                 fputil::FPType::IEEE754_Binary32,
                             int> = 0>
  LIBC_INLINE constexpr operator float() const {
    return fputil::cast<float>(*this);
  }

  template <typename T, cpp::enable_if_t<cpp::is_integral_v<T>, int> = 0>
  LIBC_INLINE constexpr explicit operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }

  LIBC_INLINE bool operator==(Float16 other) const {
    return fputil::equals(*this, other);
  }

  LIBC_INLINE bool operator!=(Float16 other) const {
    return !fputil::equals(*this, other);
  }

  LIBC_INLINE bool operator<(Float16 other) const {
    return fputil::less_than(*this, other);
  }

  LIBC_INLINE bool operator<=(Float16 other) const {
    return fputil::less_than_or_equals(*this, other);
  }

  LIBC_INLINE bool operator>(Float16 other) const {
    return fputil::greater_than(*this, other);
  }

  LIBC_INLINE bool operator>=(Float16 other) const {
    return fputil::greater_than_or_equals(*this, other);
  }

  LIBC_INLINE constexpr Float16 operator-() const {
    fputil::FPBits<float16> result(*this);
    result.set_sign(result.is_pos() ? Sign::NEG : Sign::POS);
    return result.get_val();
  }

  LIBC_INLINE Float16 operator+(Float16 other) const {
    return fputil::generic::add<Float16>(*this, other);
  }

  LIBC_INLINE Float16 operator-(Float16 other) const {
    return fputil::generic::sub<Float16>(*this, other);
  }

  LIBC_INLINE Float16 operator*(Float16 other) const {
    return fputil::generic::mul<float16>(*this, other);
  }

  LIBC_INLINE Float16 operator/(Float16 other) const {
    return fputil::generic::div<float16>(*this, other);
  }

  LIBC_INLINE Float16 &operator*=(const Float16 &other) {
    *this = *this * other;
    return *this;
  }
}; // struct Float16

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_USE_SOFT_FLOAT16

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT16_H
