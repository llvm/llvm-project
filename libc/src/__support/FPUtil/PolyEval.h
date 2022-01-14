//===-- Common header for PolyEval implementations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_POLYEVAL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_POLYEVAL_H

#include "src/__support/CPP/TypeTraits.h"
#include "src/__support/architectures.h"

// Evaluate polynomial using Horner's Scheme:
// With polyeval(x, a_0, a_1, ..., a_n) = a_n * x^n + ... + a_1 * x + a_0, we
// evaluated it as:  a_0 + x * (a_1 + x * ( ... (a_(n-1) + x * a_n) ... ) ) ).
// We will use fma instructions if available.
// Example: to evaluate x^3 + 2*x^2 + 3*x + 4, call
//   polyeval( x, 4.0, 3.0, 2.0, 1.0 )

#if defined(LLVM_LIBC_ARCH_X86_64) || defined(LLVM_LIBC_ARCH_AARCH64)
#include "FMA.h"

namespace __llvm_libc {
namespace fputil {

template <typename T> static inline T polyeval(T x, T a0) { return a0; }

template <typename T, typename... Ts>
INLINE_FMA static inline T polyeval(T x, T a0, Ts... a) {
  return fma(x, polyeval(x, a...), a0);
}

} // namespace fputil
} // namespace __llvm_libc

#ifdef LLVM_LIBC_ARCH_X86_64

// [DISABLED] There is a regression with using vectorized version for polyeval
// compared to the naive Horner's scheme with fma.  Need further investigation
// #include "x86_64/PolyEval.h"

#endif // LLVM_LIBC_ARCH_X86_64

#else

namespace __llvm_libc {
namespace fputil {

template <typename T> static inline T polyeval(T x, T a0) { return a0; }

template <typename T, typename... Ts>
static inline T polyeval(T x, T a0, Ts... a) {
  return x * polyeval(x, a...) + a0;
}

} // namespace fputil
} // namespace __llvm_libc

#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FMA_H
