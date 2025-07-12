//===-- Definition of bfloat16 data type. -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct BFloat16 {
  uint16_t bits;

  LIBC_INLINE BFloat16() = default;

  LIBC_INLINE constexpr explicit BFloat16(uint16_t bits) : bits(bits) {}

  template <typename T> LIBC_INLINE constexpr explicit BFloat16(T value) {
    if constexpr (cpp::is_floating_point_v<T>) {
      bits = fputil::cast<bfloat16>(value).bits;
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
      bits = xd.template as<bfloat16, /*ShouldSignalExceptions=*/true>().bits;

    } else {
      bits = fputil::cast<bfloat16>(static_cast<float>(value)).bits;
    }
  }

  template <cpp::enable_if_t<fputil::get_fp_type<float>() ==
                                 fputil::FPType::IEEE754_Binary32,
                             int> = 0>
  LIBC_INLINE constexpr operator float() const {
    uint32_t x_bits = static_cast<uint32_t>(bits) << 16U;
    return cpp::bit_cast<float>(x_bits);
  }
}; // struct BFloat16

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
