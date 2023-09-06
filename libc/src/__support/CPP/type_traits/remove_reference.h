//===-- remove_reference type_traits ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_REMOVE_REFERENCE_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_REMOVE_REFERENCE_H

#include "src/__support/CPP/type_traits/type_identity.h"

namespace __llvm_libc::cpp {

// remove_reference
template <class T> struct remove_reference : cpp::type_identity<T> {};
template <class T> struct remove_reference<T &> : cpp::type_identity<T> {};
template <class T> struct remove_reference<T &&> : cpp::type_identity<T> {};
template <class T>
using remove_reference_t = typename remove_reference<T>::type;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_REMOVE_REFERENCE_H
