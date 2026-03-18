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

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct Float128 {
  UInt128 bits = 0;

  constexpr Float128() = default;
  constexpr explicit Float128(UInt128 value) : bits(value) {}

  constexpr UInt128 get_bits() const { return bits; }

}; 
} // namespace fputil
}// namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FLOAT128_H
