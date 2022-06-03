//===-- Common header for PolyEval implementations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_POLYEVAL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_POLYEVAL_H

#include "multiply_add.h"

// Evaluate polynomial using Horner's Scheme:
// With polyeval(x, a_0, a_1, ..., a_n) = a_n * x^n + ... + a_1 * x + a_0, we
// evaluated it as:  a_0 + x * (a_1 + x * ( ... (a_(n-1) + x * a_n) ... ) ) ).
// We will use FMA instructions if available.
// Example: to evaluate x^3 + 2*x^2 + 3*x + 4, call
//   polyeval( x, 4.0, 3.0, 2.0, 1.0 )

namespace __llvm_libc {
namespace fputil {

template <typename T> static inline T polyeval(T x, T a0) { return a0; }

template <typename T, typename... Ts>
static inline T polyeval(T x, T a0, Ts... a) {
  return multiply_add(x, polyeval(x, a...), a0);
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_POLYEVAL_H
