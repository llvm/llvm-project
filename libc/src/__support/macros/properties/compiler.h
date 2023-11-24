//===-- Compile time compiler detection -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_COMPILER_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_COMPILER_H

#if defined(__clang__)
#define LIBC_COMPILER_IS_CLANG
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define LIBC_COMPILER_IS_GCC
#endif

#if defined(_MSC_VER)
#define LIBC_COMPILER_IS_MSC
#endif

// Check compiler features
#if defined(FLT128_MANT_DIG)
// C23 _Float128 type is available.
#define LIBC_COMPILER_HAS_FLOAT128
#define LIBC_FLOAT128_IS_C23
using float128 = _Float128;

#elif defined(__SIZEOF_FLOAT128__)
// Builtin __float128 is available.
#define LIBC_COMPILER_HAS_FLOAT128
#define LIBC_FLOAT128_IS_BUILTIN
using float128 = __float128;

#elif (defined(__linux__) && defined(__aarch64__))
// long double on Linux aarch64 is 128-bit floating point.
#define LIBC_COMPILER_HAS_FLOAT128
#define LIBC_FLOAT128_IS_LONG_DOUBLE
using float128 = long double;

#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_COMPILER_H
