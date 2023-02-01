//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_SQRT_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_SQRT_H

#include "src/__support/architectures.h"
#include "src/__support/common.h"

#if !defined(LLVM_LIBC_ARCH_AARCH64)
#error "Invalid include"
#endif

#include "src/__support/FPUtil/generic/sqrt.h"

namespace __llvm_libc {
namespace fputil {

template <> LIBC_INLINE float sqrt<float>(float x) {
  float y;
  __asm__ __volatile__("fsqrt %s0, %s1\n\t" : "=w"(y) : "w"(x));
  return y;
}

template <> LIBC_INLINE double sqrt<double>(double x) {
  double y;
  __asm__ __volatile__("fsqrt %d0, %d1\n\t" : "=w"(y) : "w"(x));
  return y;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_SQRT_H
