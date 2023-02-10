//===-- Portable attributes -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header file defines macros for declaring attributes for functions,
// types, and variables.
//
// These macros are used within llvm-libc and allow the compiler to optimize,
// where applicable, certain function calls.
//
// Most macros here are exposing GCC or Clang features, and are stubbed out for
// other compilers.

#ifndef LLVM_LIBC_SUPPORT_MACROS_ATTRIBUTES_H
#define LLVM_LIBC_SUPPORT_MACROS_ATTRIBUTES_H

#define LIBC_INLINE inline
#define LIBC_INLINE_ASM __asm__ __volatile__

// We use a template to implement likely/unlikely to make sure that we don't
// accidentally pass an integer.
namespace __llvm_libc::details {
template <typename T>
constexpr LIBC_INLINE bool expects_bool_condition(T value, T expected) {
  return __builtin_expect(value, expected);
}
} // namespace __llvm_libc::details
#define LIBC_LIKELY(x) __llvm_libc::details::expects_bool_condition(x, true)
#define LIBC_UNLIKELY(x) __llvm_libc::details::expects_bool_condition(x, false)

#define LIBC_UNUSED __attribute__((unused))

#endif // LLVM_LIBC_SUPPORT_MACROS_ATTRIBUTES_H
