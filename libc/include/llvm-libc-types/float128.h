//===-- Definition of float128 type ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_FLOAT128_H__
#define __LLVM_LIBC_TYPES_FLOAT128_H__

#if defined(__clang__)
#define LIBC_COMPILER_IS_CLANG
#define LIBC_COMPILER_CLANG_VER (__clang_major__ * 100 + __clang_minor__)
#elif defined(__GNUC__)
#define LIBC_COMPILER_IS_GCC
#define LIBC_COMPILER_GCC_VER (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

#if (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1301)) &&     \
    (defined(__aarch64__) || defined(__riscv) || defined(__x86_64__))
typedef _Float128 float128;
#elif (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 600)) &&\
    (defined(__x86_64__) && !defined(__Fuchsia__))
typedef __float128 float128;
#elif (LDBL_MANT_DIG == 113) || (__LDBL_MANT_DIG__ == 113)
typedef long double float128;
#endif

#endif // __LLVM_LIBC_TYPES_FLOAT128_H__
