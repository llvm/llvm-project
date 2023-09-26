//===-- Portable optimization macros ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header file defines portable macros for performance optimization.

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_OPTIMIZATION_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_OPTIMIZATION_H

#include "src/__support/macros/attributes.h"          // LIBC_INLINE
#include "src/__support/macros/config.h"              // LIBC_HAS_BUILTIN
#include "src/__support/macros/properties/compiler.h" // LIBC_COMPILER_IS_CLANG

// We use a template to implement likely/unlikely to make sure that we don't
// accidentally pass an integer.
namespace LIBC_NAMESPACE::details {
template <typename T>
LIBC_INLINE constexpr bool expects_bool_condition(T value, T expected) {
  return __builtin_expect(value, expected);
}
} // namespace LIBC_NAMESPACE::details
#define LIBC_LIKELY(x) LIBC_NAMESPACE::details::expects_bool_condition(x, true)
#define LIBC_UNLIKELY(x)                                                       \
  LIBC_NAMESPACE::details::expects_bool_condition(x, false)

#if defined(LIBC_COMPILER_IS_CLANG)
#define LIBC_LOOP_NOUNROLL _Pragma("nounroll")
#elif defined(LIBC_COMPILER_IS_GCC)
#define LIBC_LOOP_NOUNROLL _Pragma("GCC unroll 0")
#else
#error "Unhandled compiler"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_OPTIMIZATION_H
