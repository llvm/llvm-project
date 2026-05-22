//===-- Implementation header for SIMD sqrtf ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATHVEC_SQRTF_H
#define LLVM_LIBC_SRC___SUPPORT_MATHVEC_SQRTF_H

#include "src/__support/CPP/simd.h"
#define LIBC_MATH (LIBC_MATH_NO_ERRNO | LIBC_MATH_NO_EXCEPT)
#include "src/__support/math/sqrtf.h"

namespace LIBC_NAMESPACE_DECL {

namespace mathvec {

template <size_t N>
LIBC_INLINE cpp::simd<float, N> sqrtf(cpp::simd<float, N> x) {
  cpp::simd<float, N> result;

  for (size_t i = 0; i < N; ++i)
    result[i] = math::sqrtf(x[i]);

  return result;
}

} // namespace mathvec

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATHVEC_SQRTF_H
