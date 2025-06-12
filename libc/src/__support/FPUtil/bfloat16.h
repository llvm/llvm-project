//===-- Definition of bfloat16 data type. -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H

#include "FEnvImpl.h"
#include "FPBits.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "multiply_add.h"
#include "rounding_mode.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h" // bfloat16

#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
struct BFloat16 {
  uint16_t bits;

  BFloat16() = default;

  constexpr explicit BFloat16(uint16_t bits) : bits(bits) {}

  // fputil::cast<bfloat16>(T) only works for T = floating point type
  // TODO: integer to bfloat16 conversion
  template <typename T>
  constexpr explicit BFloat16(T x) : bits(fputil::cast<bfloat16>(x).bits) {}

  constexpr bool operator==(const BFloat16 other) const {
    return bits == other.bits;
  }

  constexpr float as_float() const {
    uint32_t x_bits = static_cast<uint32_t>(bits) << 16U;
    return cpp::bit_cast<float>(x_bits);
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
