//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_H

#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid include"
#endif

#include "src/__support/FPUtil/generic/sqrt.h"

namespace __llvm_libc {
namespace fputil {

template <> LIBC_INLINE float sqrt<float>(float x) {
  float result;
  __asm__ __volatile__("sqrtss %x1, %x0" : "=x"(result) : "x"(x));
  return result;
}

template <> LIBC_INLINE double sqrt<double>(double x) {
  double result;
  __asm__ __volatile__("sqrtsd %x1, %x0" : "=x"(result) : "x"(x));
  return result;
}

#ifdef LONG_DOUBLE_IS_DOUBLE
template <> LIBC_INLINE long double sqrt<long double>(long double x) {
  long double result;
  __asm__ __volatile__("sqrtsd %x1, %x0" : "=x"(result) : "x"(x));
  return result;
}
#else
template <> LIBC_INLINE long double sqrt<long double>(long double x) {
  __asm__ __volatile__("fsqrt" : "+t"(x));
  return x;
}
#endif

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_H
