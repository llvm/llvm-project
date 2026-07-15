//===-- 64-bit unsigned fractional type -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FRAC64_H
#define LLVM_LIBC_SRC___SUPPORT_FRAC64_H

#include "hdr/stdint_proxy.h" // uint64_t
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_INLINE
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE_DECL {

// Q0.64
struct Frac64 {
  uint64_t val;

  LIBC_INLINE constexpr Frac64() : val(0) {}
  LIBC_INLINE constexpr explicit Frac64(uint64_t value) : val(value) {}

  LIBC_INLINE constexpr Frac64 operator~() const { return Frac64(~val); }

  LIBC_INLINE constexpr Frac64 operator+(const Frac64 &other) const {
    return Frac64(val + other.val);
  }

  LIBC_INLINE constexpr Frac64 operator-(const Frac64 &other) const {
    return Frac64(val - other.val);
  }

  LIBC_INLINE constexpr Frac64 operator*(const Frac64 &other) const {
    return Frac64(
        static_cast<uint64_t>((static_cast<UInt128>(val) * other.val) >> 64));
  }

  LIBC_INLINE constexpr Frac64 &operator+=(const Frac64 &other) {
    *this = *this + other;
    return *this;
  }

  LIBC_INLINE constexpr Frac64 &operator-=(const Frac64 &other) {
    *this = *this - other;
    return *this;
  }

  LIBC_INLINE constexpr Frac64 &operator*=(const Frac64 &other) {
    *this = *this * other;
    return *this;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FRAC64_H
