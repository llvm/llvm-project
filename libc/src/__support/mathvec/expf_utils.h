//===-- Common utils for exp function ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXP_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXP_UTILS_H

#include "src/__support/CPP/simd.h"
#include "src/__support/mathvec/common_constants.h"

namespace LIBC_NAMESPACE_DECL {

template <size_t N>
LIBC_INLINE cpp::simd<double, N> exp_lookup(cpp::simd<uint64_t, N> u) {
  auto index = u & cpp::simd<uint64_t, N>(0x3f);
  auto mantissa = cpp::gather<cpp::simd<uint64_t, N>>(
      true, index, common_constants_internal::EXP_MANTISSA);
  auto exponent = (u >> 6) << 52;
  auto result = mantissa | exponent;
  return reinterpret_cast<cpp::simd<double, N>>(result);
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXP_UTILS_H
