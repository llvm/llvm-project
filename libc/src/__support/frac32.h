//===-- 32-bit unsigned fractional type -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FRAC32_H
#define LLVM_LIBC_SRC___SUPPORT_FRAC32_H

#include "src/__support/big_int.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Q0.32
struct Frac32 : public UInt<32> {
  using UInt<32>::UInt;

  LIBC_INLINE constexpr Frac32 operator~() const { return Frac32(~val[0]); }

  LIBC_INLINE constexpr Frac32 operator+(Frac32 other) const {
    return Frac32(val[0] + other.val[0]);
  }

  LIBC_INLINE constexpr Frac32 operator-(Frac32 other) const {
    return Frac32(val[0] - other.val[0]);
  }

  LIBC_INLINE constexpr Frac32 operator*(Frac32 other) const {
    UInt<32> r = UInt<32>::quick_mul_hi(UInt<32>(other));
    return Frac32(r.val[0]);
  }

  LIBC_INLINE constexpr Frac32 &operator+=(Frac32 other) {
    *this = *this + other;
    return *this;
  }

  LIBC_INLINE constexpr Frac32 &operator-=(Frac32 other) {
    *this = *this - other;
    return *this;
  }

  LIBC_INLINE constexpr Frac32 &operator*=(Frac32 other) {
    *this = *this * other;
    return *this;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FRAC32_H
