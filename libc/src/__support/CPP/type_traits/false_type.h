//===-- false_type type_traits ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_FALSE_TYPE_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_FALSE_TYPE_H

#include "src/__support/CPP/type_traits/bool_constant.h"

namespace __llvm_libc::cpp {

// false_type
using false_type = cpp::bool_constant<false>;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_FALSE_TYPE_H
