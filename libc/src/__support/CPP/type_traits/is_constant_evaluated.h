//===-- is_constant_evaluated type_traits -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_CONSTANT_EVALUATED_H

#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE::cpp {

LIBC_INLINE constexpr bool is_constant_evaluated() {
  return __builtin_is_constant_evaluated();
}

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
