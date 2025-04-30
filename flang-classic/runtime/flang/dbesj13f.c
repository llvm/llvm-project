/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	dbesj13f.c - Implements LIB3F dbesj1 subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define j1 _j1
#endif

extern double j1(double);

double ENT3F(DBESJ1, dbesj1)(double *x) { return j1(*x); }
