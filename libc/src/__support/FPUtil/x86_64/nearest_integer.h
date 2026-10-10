//===--- Round floating point to nearest integer on x86-64 ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_X86_64_NEAREST_INTEGER_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_X86_64_NEAREST_INTEGER_H

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid include"
#endif

#if !defined(__SSE4_2__)
#error "SSE4.2 instruction set is not supported"
#endif

#include <immintrin.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

// Without AVX, `roundss` and `roundsd` are 2-operand instructions that preserve
// the upper bits of the destination XMM register. Because only lane 0 of
// `_mm_round_ss(xmm, xmm, 8)` / `_mm_round_sd(xmm, xmm, 8)` is used, the
// compiler may ignore the first operand and allocate a different destination
// register, creating a false dependency on its previous value. Using inline
// assembly with `"+x"(x)` forces identical source and destination registers
// without the extra lane-zeroing instructions required by `_mm_round_ps` /
// `_mm_round_pd`.
LIBC_INLINE float nearest_integer(float x) {
#ifdef __AVX__
  __m128 xmm = _mm_set_ss(x); // NOLINT
  __m128 ymm =
      _mm_round_ss(xmm, xmm, _MM_ROUND_NEAREST | _MM_FROUND_NO_EXC); // NOLINT
  return ymm[0];
#else
  LIBC_INLINE_ASM("roundss $0x8, %0, %0" : "+x"(x));
  return x;
#endif
}

LIBC_INLINE double nearest_integer(double x) {
#ifdef __AVX__
  __m128d xmm = _mm_set_sd(x); // NOLINT
  __m128d ymm =
      _mm_round_sd(xmm, xmm, _MM_ROUND_NEAREST | _MM_FROUND_NO_EXC); // NOLINT
  return ymm[0];
#else
  LIBC_INLINE_ASM("roundsd $0x8, %0, %0" : "+x"(x));
  return x;
#endif
}

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_X86_64_NEAREST_INTEGER_H
