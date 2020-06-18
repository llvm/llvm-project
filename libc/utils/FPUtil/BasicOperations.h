//===-- Basic operations on floating point numbers --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_BASIC_OPERATIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_BASIC_OPERATIONS_H

#include "FPBits.h"

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T abs(T x) {
  FPBits<T> bits(x);
  bits.sign = 0;
  return T(bits);
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_BASIC_OPERATIONS_H
