//===-- type_identity type_traits -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_TYPE_IDENTITY_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_TYPE_IDENTITY_H

namespace __llvm_libc::cpp {

// type_identity
template <typename T> struct type_identity {
  using type = T;
};

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_TYPE_IDENTITY_H
