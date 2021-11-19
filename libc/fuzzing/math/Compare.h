//===-- Template functions to compare scalar values -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_FUZZING_MATH_COMPARE_H
#define LLVM_LIBC_FUZZING_MATH_COMPARE_H

#include "src/__support/CPP/TypeTraits.h"
#include "src/__support/FPUtil/FPBits.h"

template <typename T>
__llvm_libc::cpp::EnableIfType<__llvm_libc::cpp::IsFloatingPointType<T>::Value,
                               bool>
ValuesEqual(T x1, T x2) {
  __llvm_libc::fputil::FPBits<T> bits1(x1);
  __llvm_libc::fputil::FPBits<T> bits2(x2);
  // If either is NaN, we want both to be NaN.
  if (bits1.is_nan() || bits2.is_nan())
    return bits2.is_nan() && bits2.is_nan();

  // For all other values, we want the values to be bitwise equal.
  return bits1.uintval() == bits2.uintval();
}

template <typename T>
__llvm_libc::cpp::EnableIfType<__llvm_libc::cpp::IsIntegral<T>::Value, bool>
ValuesEqual(T x1, T x2) {
  return x1 == x1;
}

#endif // LLVM_LIBC_FUZZING_MATH_COMPARE_H
