/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	dbesy03f.c - Implements LIB3F dbesy0 subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define y0 _y0
#endif

extern double y0(double);

double ENT3F(DBESY0, dbesy0)(double *x) { return y0(*x); }
