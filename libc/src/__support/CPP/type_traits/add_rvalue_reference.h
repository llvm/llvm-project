//===-- add_rvalue_reference type_traits ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_ADD_RVALUE_REFERENCE_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_ADD_RVALUE_REFERENCE_H

#include "src/__support/CPP/type_traits/type_identity.h"

namespace LIBC_NAMESPACE::cpp {

// add_rvalue_reference
namespace detail {
template <class T>
auto try_add_rvalue_reference(int) -> cpp::type_identity<T &&>;
template <class T> auto try_add_rvalue_reference(...) -> cpp::type_identity<T>;
} // namespace detail
template <class T>
struct add_rvalue_reference : decltype(detail::try_add_rvalue_reference<T>(0)) {
};
template <class T>
using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_ADD_RVALUE_REFERENCE_H
