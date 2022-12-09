//===-- Utilities for pairs of numbers. -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_NAMED_PAIR_H
#define LLVM_LIBC_SRC_SUPPORT_NAMED_PAIR_H

#define DEFINE_NAMED_PAIR_TEMPLATE(Name, FirstField, SecondField)              \
  template <typename T1, typename T2 = T1> struct Name {                       \
    T1 FirstField;                                                             \
    T2 SecondField;                                                            \
  }

#endif // LLVM_LIBC_SRC_SUPPORT_NAMED_PAIR_H
