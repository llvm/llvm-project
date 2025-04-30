/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	dbesy13f.c - Implements LIB3F dbesy1 subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define y1 _y1
#endif

extern double y1(double);

double ENT3F(DBESY1, dbesy1)(double *x) { return y1(*x); }
