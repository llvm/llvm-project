//===-- bool_constant type_traits -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_BOOL_CONSTANT_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_BOOL_CONSTANT_H

#include "src/__support/CPP/type_traits/integral_constant.h"

namespace LIBC_NAMESPACE::cpp {

// bool_constant
template <bool V> using bool_constant = cpp::integral_constant<bool, V>;

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_BOOL_CONSTANT_H
