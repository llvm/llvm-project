/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	besy13f.c - Implements LIB3F besy1 subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define y1 _y1
#endif

extern double y1(double);

float ENT3F(BESY1, besy1)(float *x) { return (float)y1(*x); }
