//===-- Implementation header for SIMD expf ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXPF_H
#define LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXPF_H

#include "src/__support/CPP/simd.h"
#include "src/__support/math/expf.h"

namespace LIBC_NAMESPACE_DECL {

namespace mathvec {

template <size_t N>
LIBC_INLINE static constexpr cpp::simd<float, N> expf(cpp::simd<float, N> x) {
  cpp::simd<float, N> ret = 0.0f;

  for (size_t i = 0; i < N; i++)
    ret[i] = math::expf(x[i]);

  return ret;
}

} // namespace mathvec

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXPF_H
