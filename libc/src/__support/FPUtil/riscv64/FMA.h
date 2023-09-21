//===-- RISCV64 implementations of the fma function -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_RISCV64_FMA_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_RISCV64_FMA_H

#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA

#if !defined(LIBC_TARGET_ARCH_IS_RISCV64)
#error "Invalid include"
#endif

#if !defined(LIBC_TARGET_CPU_HAS_FMA)
#error "FMA instructions are not supported"
#endif

#include "src/__support/CPP/type_traits.h"

namespace __llvm_libc {
namespace fputil {

template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_same_v<T, float>, T> fma(T x, T y, T z) {
  float result;
  LIBC_INLINE_ASM("fmadd.s %0, %1, %2, %3\n\t"
                  : "=f"(result)
                  : "f"(x), "f"(y), "f"(z));
  return result;
}

template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_same_v<T, double>, T> fma(T x, T y, T z) {
  double result;
  LIBC_INLINE_ASM("fmadd.d %0, %1, %2, %3\n\t"
                  : "=f"(result)
                  : "f"(x), "f"(y), "f"(z));
  return result;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_RISCV64_FMA_H
