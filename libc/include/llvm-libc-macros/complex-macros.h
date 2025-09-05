//===-- Definition of macros to be used with complex functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_COMPLEX_MACROS_H
#define __LLVM_LIBC_MACROS_COMPLEX_MACROS_H

#include "cfloat128-macros.h"
#include "cfloat16-macros.h"

#ifndef __STDC_NO_COMPLEX__

#define __STDC_VERSION_COMPLEX_H__ 202311L

#define complex _Complex
#define _Complex_I ((_Complex float)1.0fi)
#define I _Complex_I

// TODO: Add imaginary macros once GCC or Clang support _Imaginary builtin-type.

#define CMPLX(x, y) __builtin_complex((double)(x), (double)(y))
#define CMPLXF(x, y) __builtin_complex((float)(x), (float)(y))
#define CMPLXL(x, y) __builtin_complex((long double)(x), (long double)(y))

#ifdef LIBC_TYPES_HAS_CFLOAT16
#if !defined(__clang__) || (__clang_major__ >= 22 && __clang_minor__ > 0)
#define CMPLXF16(x, y) __builtin_complex((_Float16)(x), (_Float16)(y))
#else
#define CMPLXF16(x, y)                                                         \
  ((complex _Float16)(__builtin_complex((float)(x), (float)(y))))
#endif
#endif // LIBC_TYPES_HAS_CFLOAT16

#ifdef LIBC_TYPES_HAS_CFLOAT128
#define CMPLXF128(x, y) __builtin_complex((float128)(x), (float128)(y))
#endif // LIBC_TYPES_HAS_CFLOAT128

#endif // __STDC_NO_COMPLEX__

#endif // __LLVM_LIBC_MACROS_COMPLEX_MACROS_H
