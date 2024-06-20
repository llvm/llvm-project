//===-- Utilities for pairs of numbers. -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_NUMBER_PAIR_H
#define LLVM_LIBC_SRC___SUPPORT_NUMBER_PAIR_H

#include "CPP/type_traits.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {

template <typename T> struct NumberPair {
  T lo = T(0);
  T hi = T(0);
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_NUMBER_PAIR_H
