//===-- move utility --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_UTILITY_MOVE_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_UTILITY_MOVE_H

#include "src/__support/CPP/type_traits/remove_reference.h"

namespace __llvm_libc::cpp {

// move
template <class T> constexpr cpp::remove_reference_t<T> &&move(T &&t) {
  return static_cast<typename cpp::remove_reference_t<T> &&>(t);
}

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_UTILITY_MOVE_H
