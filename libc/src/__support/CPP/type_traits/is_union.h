//===-- is_union type_traits ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_UNION_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_UNION_H

#include "src/__support/CPP/type_traits/bool_constant.h"
#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE::cpp {

// is_union
template <class T> struct is_union : bool_constant<__is_union(T)> {};
template <typename T>
LIBC_INLINE_VAR constexpr bool is_union_v = is_union<T>::value;

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_UNION_H
