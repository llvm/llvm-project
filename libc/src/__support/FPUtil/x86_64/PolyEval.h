//===-- Optimized PolyEval implementations for x86_64 --------- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_POLYEVAL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_POLYEVAL_H

#include "src/__support/architectures.h"
#include "src/__support/common.h"

#if !defined(LLVM_LIBC_ARCH_X86_64)
#error "Invalid include"
#endif

#include <immintrin.h>

namespace __llvm_libc {
namespace fputil {

// Cubic polynomials:
//   polyeval(x, a0, a1, a2, a3) = a3*x^3 + a2*x^2 + a1*x + a0
template <>
LIBC_INLINE float polyeval(float x, float a0, float a1, float a2, float a3) {
  __m128 xmm = _mm_set1_ps(x);                 // NOLINT
  __m128 a13 = _mm_set_ps(0.0f, x, a3, a1);    // NOLINT
  __m128 a02 = _mm_set_ps(0.0f, 0.0f, a2, a0); // NOLINT
  // r = (0, x^2, a3*x + a2, a1*x + a0)
  __m128 r = _mm_fmadd_ps(a13, xmm, a02); // NOLINT
  // result = (a3*x + a2) * x^2 + (a1*x + a0)
  return fma(r[2], r[1], r[0]);
}

template <>
LIBC_INLINE double polyeval(double x, double a0, double a1, double a2,
                            double a3) {
  __m256d xmm = _mm256_set1_pd(x);               // NOLINT
  __m256d a13 = _mm256_set_pd(0.0, x, a3, a1);   // NOLINT
  __m256d a02 = _mm256_set_pd(0.0, 0.0, a2, a0); // NOLINT
  // r = (0, x^2, a3*x + a2, a1*x + a0)
  __m256d r = _mm256_fmadd_pd(a13, xmm, a02); // NOLINT
  // result = (a3*x + a2) * x^2 + (a1*x + a0)
  return fma(r[2], r[1], r[0]);
}

// Quintic polynomials:
//   polyeval(x, a0, a1, a2, a3, a4, a5) = a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 +
//                                         + a1*x + a0
template <>
LIBC_INLINE float polyeval(float x, float a0, float a1, float a2, float a3,
                           float a4, float a5) {
  __m128 xmm = _mm_set1_ps(x);                 // NOLINT
  __m128 a25 = _mm_set_ps(0.0f, x, a5, a2);    // NOLINT
  __m128 a14 = _mm_set_ps(0.0f, 0.0f, a4, a1); // NOLINT
  __m128 a03 = _mm_set_ps(0.0f, 0.0f, a3, a0); // NOLINT
  // r1 = (0, x^2, a5*x + a4, a2*x + a1)
  __m128 r1 = _mm_fmadd_ps(a25, xmm, a14); // NOLINT
  // r2 = (0, x^3, (a5*x + a4)*x + a3, (a2*x + a1)*x + a0
  __m128 r2 = _mm_fmadd_ps(r1, xmm, a03); // NOLINT
  // result = ((a5*x + a4)*x + a3) * x^3 + ((a2*x + a1)*x + a0)
  return fma(r2[2], r2[1], r2[0]);
}

template <>
LIBC_INLINE double polyeval(double x, double a0, double a1, double a2,
                            double a3, double a4, double a5) {
  __m256d xmm = _mm256_set1_pd(x);               // NOLINT
  __m256d a25 = _mm256_set_pd(0.0, x, a5, a2);   // NOLINT
  __m256d a14 = _mm256_set_pd(0.0, 0.0, a4, a1); // NOLINT
  __m256d a03 = _mm256_set_pd(0.0, 0.0, a3, a0); // NOLINT
  // r1 = (0, x^2, a5*x + a4, a2*x + a1)
  __m256d r1 = _mm256_fmadd_pd(a25, xmm, a14); // NOLINT
  // r2 = (0, x^3, (a5*x + a4)*x + a3, (a2*x + a1)*x + a0
  __m256d r2 = _mm256_fmadd_pd(r1, xmm, a03); // NOLINT
  // result = ((a5*x + a4)*x + a3) * x^3 + ((a2*x + a1)*x + a0)
  return fma(r2[2], r2[1], r2[0]);
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_POLYEVAL_H
