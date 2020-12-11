//===-- Utils for abs and friends -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H

namespace __llvm_libc {

template <typename T> static inline T integer_abs(T n) {
  if (n < 0)
    return -n;
  return n;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H
