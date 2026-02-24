//===-- floatunoidf.c - Implement __floatunoidf ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __floatunoidf for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

#define SRC_U256
#define DST_DOUBLE
#include "int_to_fp_impl.inc"

COMPILER_RT_ABI double __floatunoidf(ou_int a) { return __floatXiYf__(a); }

#endif // CRT_HAS_256BIT
