//===-- modoi3.c - Implement __modoi3 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __modoi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a % b

#define fixint_t oi_int
#define fixuint_t ou_int
#define ASSIGN_UMOD(res, a, b) __udivmodoi4((a), (b), &(res))
#include "int_div_impl.inc"

COMPILER_RT_ABI oi_int __modoi3(oi_int a, oi_int b) { return __modXi3(a, b); }

#endif // CRT_HAS_256BIT
