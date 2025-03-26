//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_CLCMACROS_H__
#define __CLC_CLCMACROS_H__

/* 6.9 Preprocessor Directives and Macros
 * Some of these are handled by clang or passed by clover */
#if __OPENCL_VERSION__ >= 110
#define CLC_VERSION_1_0 100
#define CLC_VERSION_1_1 110
#endif

#if __OPENCL_VERSION__ >= 120
#define CLC_VERSION_1_2 120
#endif

#define NULL ((void *)0)

#define __kernel_exec(X, typen)                                                \
  __kernel __attribute__((work_group_size_hint(X, 1, 1)))                      \
  __attribute__((vec_type_hint(typen)))

#define kernel_exec(X, typen) __kernel_exec(X, typen)

#endif // __CLC_CLCMACROS_H__
