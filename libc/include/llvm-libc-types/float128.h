//===-- Definition of float128 type ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_FLOAT128_H__
#define __LLVM_LIBC_TYPES_FLOAT128_H__

#include <llvm-libc-macros/float-macros.h> // LDBL_MANT_DIG

// Define temporary compiler and its version
#if defined(__clang__)
#define __LIBC_COMPILER_IS_CLANG
#define __LIBC_COMPILER_CLANG_VER (__clang_major__ * 100 + __clang_minor__)
#elif defined(__GNUC__)
#define __LIBC_COMPILER_IS_GCC
#define __LIBC_COMPILER_GCC_VER (__GNUC__ * 100 + __GNUC_MINOR__)
#endif // __clang__, __GNUC__

#if (defined(__LIBC_COMPILER_GCC_VER) && (__LIBC_COMPILER_GCC_VER >= 1301)) && \
    (defined(__aarch64__) || defined(__riscv) || defined(__x86_64__))
#define LIBC_COMPILER_HAS_C23_FLOAT128
typedef _Float128 float128;
#elif (defined(__LIBC_COMPILER_CLANG_VER) &&                                   \
       (__LIBC_COMPILER_CLANG_VER >= 600)) &&                                  \
    (defined(__x86_64__) && !defined(__Fuchsia__))
#define LIBC_COMPILER_HAS_FLOAT128_EXTENSION
typedef __float128 float128;
#elif (LDBL_MANT_DIG == 113)
typedef long double float128;
#endif

// Clean up temporary macros
#ifdef __LIBC_COMPILER_IS_CLANG
#undef __LIBC_COMPILER_IS_CLANG
#undef __LIBC_COMPILER_CLANG_VER
#elif defined(__LIBC_COMPILER_IS_GCC)
#undef __LIBC_COMPILER_IS_GCC
#undef __LIBC_COMPILER_GCC_VER
#endif // __LIBC_COMPILER_IS_CLANG, __LIBC_COMPILER_IS_GCC

#endif // __LLVM_LIBC_TYPES_FLOAT128_H__
