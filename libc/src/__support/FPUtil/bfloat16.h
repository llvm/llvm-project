//===-- Definition of bfloat16 data type. -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H

#include "src/__support/CPP/bit.h"                 // cpp::bit_cast
#include "src/__support/CPP/type_traits.h"         // cpp::is_floating_point_v
#include "src/__support/FPUtil/cast.h"             // fputil::cast
#include "src/__support/macros/config.h"           // LIBC_NAMESPACE_DECL
#include "src/__support/macros/properties/types.h" // bfloat16

#include <stdint.h> // uint16_t

namespace LIBC_NAMESPACE_DECL {
struct BFloat16 {
  uint16_t bits;

  BFloat16() = default;

  constexpr explicit BFloat16(uint16_t bits) : bits(bits) {}

  template <typename T> constexpr explicit BFloat16(T value) {
    if constexpr (cpp::is_floating_point_v<T>) {
      bits = fputil::cast<bfloat16>(value).bits;
    } else if constexpr (cpp::is_integral_v<T>) {
      //  TODO: verify this
      Sign sign = Sign::POS;

      if constexpr (cpp::is_signed_v<T>) {
        if (value < 0) {
          sign = Sign::NEG;
          value = -value;
        }
      }
      cpp::make_unsigned_t<T> uvalue = value;

      int msb = cpp::bit_width(uvalue) - 1;
      int biased_exponent = msb + 127;

      if (biased_exponent > 255) {
        bits = fputil::FPBits<bfloat16>::inf(sign).get_val().bits;
        return;
      }

      // TODO: currently, this is just truncation. need to handle rounding.
      if (msb >= 7) {
        uvalue >>= (msb - 7);
      } else {
        uvalue <<= (7 - msb);
      }

      const auto exp = static_cast<uint16_t>(biased_exponent << 7U);
      const uint16_t mant = static_cast<uint16_t>(uvalue) & 0x7F;

      bits = exp | mant;
      if (sign.is_neg()) {
        bits |= (1U << 15);
      }
    } else {
      bits = fputil::cast<bfloat16>(static_cast<float>(value)).bits;
    }
  }

  constexpr float as_float() const {
    uint32_t x_bits = static_cast<uint32_t>(bits) << 16U;
    return cpp::bit_cast<float>(x_bits);
  }
}; // struct BFloat16

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
