//===-- is_object type_traits -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_IS_OBJECT_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_IS_OBJECT_H

#include "src/__support/CPP/type_traits/bool_constant.h"
#include "src/__support/CPP/type_traits/is_array.h"
#include "src/__support/CPP/type_traits/is_class.h"
#include "src/__support/CPP/type_traits/is_scalar.h"
#include "src/__support/CPP/type_traits/is_union.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc::cpp {

// is_object
template <class T>
struct is_object
    : cpp::bool_constant<cpp::is_scalar_v<T> || cpp::is_array_v<T> ||
                         cpp::is_union_v<T> || cpp::is_class_v<T>> {};
template <class T>
LIBC_INLINE_VAR constexpr bool is_object_v = is_object<T>::value;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_IS_OBJECT_H
