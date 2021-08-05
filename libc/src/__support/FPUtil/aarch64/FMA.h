//===-- Aarch64 implementations of the fma function -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FMA_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FMA_H

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

template <typename T>
cpp::EnableIfType<cpp::IsSame<T, float>::Value, T> fma(T x, T y, T z) {
  float result;
  __asm__ __volatile__("fmadd %s0, %s1, %s2, %s3\n\t"
                       : "=w"(result)
                       : "w"(x), "w"(y), "w"(z));
  return result;
}

template <typename T>
cpp::EnableIfType<cpp::IsSame<T, double>::Value, T> fma(T x, T y, T z) {
  double result;
  __asm__ __volatile__("fmadd %d0, %d1, %d2, %d3\n\t"
                       : "=w"(result)
                       : "w"(x), "w"(y), "w"(z));
  return result;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FMA_H
