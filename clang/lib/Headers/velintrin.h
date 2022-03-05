/*===---- velintrin.h - VEL intrinsics for VE ------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __VEL_INTRIN_H__
#define __VEL_INTRIN_H__

// Vector registers
typedef double __vr __attribute__((__vector_size__(2048)));

// TODO: Vector mask registers
// Depend on https://reviews.llvm.org/D88905
#if 0
#if __STDC_VERSION__ >= 199901L
// For C99
typedef _Bool __vm    __attribute__((ext_vector_type(256)));
typedef _Bool __vm256 __attribute__((ext_vector_type(256)));
typedef _Bool __vm512 __attribute__((ext_vector_type(512)));
#else
#ifdef __cplusplus
// For C++
typedef bool __vm    __attribute__((ext_vector_type(256)));
typedef bool __vm256 __attribute__((ext_vector_type(256)));
typedef bool __vm512 __attribute__((ext_vector_type(512)));
#else
#error need C++ or C99 to use vector intrinsics for VE
#endif
#endif
#endif

// Use generated intrinsic name definitions
#include <velintrin_gen.h>

#endif
