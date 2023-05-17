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
#define LIBC_UNUSED __attribute__((unused))

#endif // LLVM_LIBC_SUPPORT_MACROS_ATTRIBUTES_H
