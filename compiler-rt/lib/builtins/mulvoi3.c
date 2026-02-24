//===-- mulvoi3.c - Implement __mulvoi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __mulvoi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a * b

// Effects: aborts if a * b overflows

#define fixint_t oi_int
#define fixuint_t ou_int
#include "int_mulv_impl.inc"

COMPILER_RT_ABI oi_int __mulvoi3(oi_int a, oi_int b) { return __mulvXi3(a, b); }

#endif // CRT_HAS_256BIT
