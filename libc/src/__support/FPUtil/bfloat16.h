//===-- Utilities for bfloat16 data type. -----------------------*- C++ -*-===//
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
#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"

#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct BFloat16 {
  uint16_t bits;

  constexpr BFloat16() = default;

  constexpr BFloat16(float x) {
    FPBits x_bits(x);

    uint32_t val = x_bits.get_val();
    bits = fputil::cast<BFloat16>(x);
  }

  constexpr float as_float() const {
    uint32_t val = static_cast<uint32_t>(bits) << 16U;
    return cpp::bit_cast<float>(val);
  }
}

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_BFLOAT16_H
