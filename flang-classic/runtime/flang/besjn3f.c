/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	besjn3f.c - Implements LIB3F besjn subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define jn _jn
#endif

extern double jn(int, double);

float ENT3F(BESJN, besjn)(int *n, float *x) { return (float)jn(*n, *x); }
