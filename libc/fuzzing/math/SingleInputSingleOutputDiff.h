//===-- Template to diff single-input-single-output functions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_FUZZING_MATH_SINGLE_INPUT_SINGLE_OUTPUT_DIFF_H
#define LLVM_LIBC_FUZZING_MATH_SINGLE_INPUT_SINGLE_OUTPUT_DIFF_H

#include "fuzzing/math/Compare.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"

#include <stddef.h>
#include <stdint.h>

template <typename T> using SingleInputSingleOutputFunc = T (*)(T);

template <typename T>
void SingleInputSingleOutputDiff(SingleInputSingleOutputFunc<T> func1,
                                 SingleInputSingleOutputFunc<T> func2,
                                 const uint8_t *data, size_t size) {
  if (size < sizeof(T))
    return;

  T x = *reinterpret_cast<const T *>(data);

  T result1 = func1(x);
  T result2 = func2(x);

  if (!ValuesEqual(result1, result2))
    __builtin_trap();
}

template <typename T1, typename T2>
using SingleInputSingleOutputWithSideEffectFunc = T1 (*)(T1, T2 *);

template <typename T1, typename T2>
void SingleInputSingleOutputWithSideEffectDiff(
    SingleInputSingleOutputWithSideEffectFunc<T1, T2> func1,
    SingleInputSingleOutputWithSideEffectFunc<T1, T2> func2,
    const uint8_t *data, size_t size) {
  if (size < sizeof(T1))
    return;

  T1 x = *reinterpret_cast<const T1 *>(data);
  T2 sideEffect1 = 0, sideEffect2 = 0;

  T1 result1 = func1(x, &sideEffect1);
  T1 result2 = func2(x, &sideEffect2);

  if (!ValuesEqual(result1, result2))
    __builtin_trap();

  // If x is inf or nan and T2 is integral (e.g. frexp), the value of sideEffect
  // (*exp) is unspecified by the ISO C standard (§7.12.6.4).
  if (LIBC_NAMESPACE::cpp::is_floating_point_v<T1> &&
      LIBC_NAMESPACE::fputil::FPBits<T1>(x).is_inf_or_nan() &&
      LIBC_NAMESPACE::cpp::is_integral_v<T2>)
    return;

  if (!ValuesEqual(sideEffect1, sideEffect2))
    __builtin_trap();
}

#endif // LLVM_LIBC_FUZZING_MATH_SINGLE_INPUT_SINGLE_OUTPUT_DIFF_H
