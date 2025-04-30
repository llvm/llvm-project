/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	dbesyn3f.c - Implements LIB3F dbesyn subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define yn _yn
#endif

extern double yn(int, double);

double ENT3F(DBESYN, dbesyn)(int *n, double *x) { return yn(*n, *x); }
