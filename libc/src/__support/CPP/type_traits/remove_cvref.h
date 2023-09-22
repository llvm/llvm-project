//===-- remove_cvref type_traits --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_REMOVE_CVREF_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_REMOVE_CVREF_H

#include "src/__support/CPP/type_traits/remove_cv.h"
#include "src/__support/CPP/type_traits/remove_reference.h"

namespace __llvm_libc::cpp {

// remove_cvref
template <typename T> struct remove_cvref {
  using type = remove_cv_t<remove_reference_t<T>>;
};
template <typename T> using remove_cvref_t = typename remove_cvref<T>::type;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_REMOVE_CVREF_H
