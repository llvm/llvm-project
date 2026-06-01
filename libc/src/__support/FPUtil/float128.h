//===-- Definition for EFloat128 data type  ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H

#include "src/__support/CPP/bit.h" // cpp::bit_cast
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/uint128.h"
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct EFloat128 {
  UInt128 bits;

  LIBC_INLINE EFloat128() = default;

  LIBC_INLINE constexpr explicit EFloat128(double x) {
    uint64_t d = cpp::bit_cast<uint64_t>(x);
    bits = static_cast<UInt128>(d) << 64U;
  }

  LIBC_INLINE constexpr explicit operator double() const {
    return cpp::bit_cast<double>(static_cast<uint64_t>(bits >> 64U));
  }

  LIBC_INLINE constexpr bool operator==(EFloat128 other) const {
    return bits == other.bits;
  }
  LIBC_INLINE constexpr bool operator!=(EFloat128 other) const {
    return bits != other.bits;
  }
};

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H
