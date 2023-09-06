//===-- is_null_pointer type_traits -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_IS_NULL_POINTER_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_IS_NULL_POINTER_H

#include "src/__support/CPP/type_traits/is_same.h"
#include "src/__support/CPP/type_traits/remove_cv.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc::cpp {

// is_null_pointer
using nullptr_t = decltype(nullptr);
template <class T>
struct is_null_pointer : cpp::is_same<cpp::nullptr_t, cpp::remove_cv_t<T>> {};

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_IS_NULL_POINTER_H
