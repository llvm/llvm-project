//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_H

#include "src/__support/architectures.h"

#if !defined(LLVM_LIBC_ARCH_X86)
#error "Invalid include"
#endif

#include "src/__support/FPUtil/generic/sqrt.h"

namespace __llvm_libc {
namespace fputil {

template <> inline float sqrt<float>(float x) {
  float result;
  __asm__ __volatile__("sqrtss %x1, %x0" : "=x"(result) : "x"(x));
  return result;
}

template <> inline double sqrt<double>(double x) {
  double result;
  __asm__ __volatile__("sqrtsd %x1, %x0" : "=x"(result) : "x"(x));
  return result;
}

template <> inline long double sqrt<long double>(long double x) {
  long double result;
  __asm__ __volatile__("fsqrt" : "=t"(result) : "t"(x));
  return result;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_H
