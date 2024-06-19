//===-- aligned_storage type_traits    --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_ALIGNED_STORAGE_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_ALIGNED_STORAGE_H

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE::cpp {

template <size_t Len, size_t Align> struct aligned_storage {
  struct type {
    alignas(Align) unsigned char data[Len];
  };
};

template <size_t Len, size_t Align>
using aligned_storage_t = typename aligned_storage<Len, Align>::type;

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_ALIGNED_STORAGE_H
