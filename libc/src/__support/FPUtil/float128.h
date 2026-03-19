//===-- Float128 software wrapper ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a minimal software-backed Float128 wrapper type used when
// the host compiler does not provide a native 128-bit floating-point type.
// The wrapper currently only stores the raw 128-bit representation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FLOAT128_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FLOAT128_H

#include "src/__support/uint128.h"
#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/FPUtil/generic/div.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/FPUtil/cast.h"

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct Float128 {
  UInt128 bits = 0;

  constexpr Float128() = default;
  constexpr explicit Float128(UInt128 value) : bits(value) {}
  constexpr explicit Float128(float128 v) : bits(cpp::bit_cast<UInt128>(v)) {} //add constructor
  constexpr UInt128 get_bits() const { return bits;}

  //basic arithmetic operators
  constexpr LIBC_INLINE float128 operator+(const Float128 &other) const {
    float128 a = cpp::bit_cast<float128>(bits);
    float128 b = cpp::bit_cast<float128>(other.bits);
    return a + b;
  }

  constexpr LIBC_INLINE float128 operator-(const Float128 &other) const {
    float128 a = cpp::bit_cast<float128>(bits);
    float128 b = cpp::bit_cast<float128>(other.bits);
    return a - b;
  }

  constexpr LIBC_INLINE float128 operator*(const Float128 &other) const {
    float128 a = cpp::bit_cast<float128>(bits);
    float128 b = cpp::bit_cast<float128>(other.bits);
    return a * b;
  }

  constexpr LIBC_INLINE float128 operator/(const Float128 &other) const {
    float128 a = cpp::bit_cast<float128>(bits);
    float128 b = cpp::bit_cast<float128>(other.bits);
    return a / b;
  }
}; 
} // namespace fputil
}// namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FLOAT128_H
