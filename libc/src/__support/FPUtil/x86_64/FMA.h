//===-- x86_64 implementations of the fma function --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FMA_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FMA_H

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

template <typename T>
static inline cpp::EnableIfType<cpp::IsSame<T, float>::Value, T> fma(T x, T y,
                                                                     T z) {
  float result = x;
  __asm__ __volatile__("vfmadd213ss %x2, %x1, %x0"
                       : "+x"(result)
                       : "x"(y), "x"(z));
  return result;
}

template <typename T>
static inline cpp::EnableIfType<cpp::IsSame<T, double>::Value, T> fma(T x, T y,
                                                                      T z) {
  double result = x;
  __asm__ __volatile__("vfmadd213sd %x2, %x1, %x0"
                       : "+x"(result)
                       : "x"(y), "x"(z));
  return result;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FMA_H
