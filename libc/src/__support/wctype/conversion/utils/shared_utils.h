//===-- Internal shared utils for wctype conversion code --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_UTILS_H

#include "hdr/stdint_proxy.h"
#include "slice.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace conversion_utils {

// Multiplies two 64-bit unsigned integers and returns the high 64 bits
LIBC_INLINE static constexpr uint64_t mul_high(uint64_t a, uint64_t b) {
  return (static_cast<UInt128>(a) * static_cast<UInt128>(b)) >> 64;
}

// Wrapping multiplication for integral types
template <typename T> LIBC_INLINE static constexpr T wrapping_mul(T a, T b) {
  static_assert(cpp::is_integral_v<T>, "wrapping_mul requires integral type");

  T result = 0;

  while (b != 0) {
    if (b & 1) {
      result = static_cast<T>(result + a);
    }
    a = static_cast<T>(a << 1);
    b = static_cast<cpp::make_unsigned_t<T>>(b) >> 1;
  }

  return result;
}

} // namespace conversion_utils

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_UTILS_H
