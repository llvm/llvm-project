//===-- remove_all_extents type_traits --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_REMOVE_ALL_EXTENTS_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_REMOVE_ALL_EXTENTS_H

#include "src/__support/macros/config.h"

namespace __llvm_libc::cpp {

// remove_all_extents
#if LIBC_HAS_BUILTIN(__remove_all_extents)
template <typename T> using remove_all_extents_t = __remove_all_extents(T);
#else
template <typename T> struct remove_all_extents {
  using type = T;
};
template <typename T> struct remove_all_extents<T[]> {
  using type = typename remove_all_extents<T>::type;
};
template <typename T, size_t _Np> struct remove_all_extents<T[_Np]> {
  using type = typename remove_all_extents<T>::type;
};
#endif
template <typename T>
using remove_all_extents_t = typename remove_all_extents<T>::type;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_REMOVE_ALL_EXTENTS_H
