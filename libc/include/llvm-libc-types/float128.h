//===-- Definition of float128 type ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_FLOAT128_H__
#define __LLVM_LIBC_TYPES_FLOAT128_H__

#include <include/llvm-libc-macros/float-macros.h> // LDBL_MANT_DIG

// Currently, C23 `_Float128` type is only defined as a built-in type in GCC 7
// or later, and only for C.  For C++, or for clang, `__float128` is defined
// instead, and only on x86-64 targets.
//
// TODO: Update C23 `_Float128` type detection again when clang supports it.
//   https://github.com/llvm/llvm-project/issues/80195
#if defined(__STDC_IEC_60559_BFP__) && !defined(__clang__) &&                  \
    !defined(__cplusplus)
// Use _Float128 C23 type.
#define LIBC_COMPILER_HAS_C23_FLOAT128
typedef _Float128 float128;
#elif defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
// Use __float128 type.  gcc and clang sometime use __SIZEOF_FLOAT128__ to
// notify the availability of __float128.
// clang also uses __FLOAT128__ macro to notify the availability of __float128
// type: https://reviews.llvm.org/D15120
#define LIBC_COMPILER_HAS_FLOAT128_EXTENSION
typedef __float128 float128;
#elif (LDBL_MANT_DIG == 113)
// Use long double.
typedef long double float128;
#endif

#endif // __LLVM_LIBC_TYPES_FLOAT128_H__
