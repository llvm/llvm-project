//===-- declval utility -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_DECLVAL_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_DECLVAL_H

#include "src/__support/CPP/type_traits/add_rvalue_reference.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc::cpp {

// declval
namespace detail {
template <typename T> LIBC_INLINE_VAR constexpr bool always_false = false;
}

template <typename T> cpp::add_rvalue_reference_t<T> declval() {
  static_assert(detail::always_false<T>,
                "declval not allowed in an evaluated context");
}

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_DECLVAL_H
