//===-- A self contained equivalent of <algorithm> --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file is minimalist on purpose but can receive a few more function if
// they prove useful.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_ALGORITHM_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_ALGORITHM_H

#include "src/__support/macros/attributes.h" // LIBC_INLINE

namespace __llvm_libc {
namespace cpp {

template <class T> LIBC_INLINE constexpr const T &max(const T &a, const T &b) {
  return (a < b) ? b : a;
}

template <class T> LIBC_INLINE constexpr const T &min(const T &a, const T &b) {
  return (a < b) ? a : b;
}

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_ALGORITHM_H
