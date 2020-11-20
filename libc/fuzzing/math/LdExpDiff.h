//===-- Template for diffing ldexp results ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/FPUtil/FPBits.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

template <typename T> using LdExpFunc = T (*)(T, int);

template <typename T>
void LdExpDiff(LdExpFunc<T> func1, LdExpFunc<T> func2, const uint8_t *data,
               size_t size) {
  constexpr size_t typeSize = sizeof(T);
  if (size < typeSize + sizeof(int))
    return;

  T x = *reinterpret_cast<const T *>(data);
  T i = *reinterpret_cast<const int *>(data + typeSize);

  T result1 = func1(x, i);
  T result2 = func2(x, i);
  if (isnan(result1)) {
    if (!isnan(result2))
      __builtin_trap();
    return;
  }
  if (isinf(result1)) {
    if (isinf(result2) != isinf(result1))
      __builtin_trap();
    return;
  }

  __llvm_libc::fputil::FPBits<T> bits1(result1);
  __llvm_libc::fputil::FPBits<T> bits2(result2);
  if (bits1.bitsAsUInt() != bits2.bitsAsUInt())
    __builtin_trap();
}
