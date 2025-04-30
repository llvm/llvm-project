/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	dbesj03f.c - Implements LIB3F dbesj0 subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define j0 _j0
#endif

extern double j0(double);

double ENT3F(DBESJ0, dbesj0)(double *x) { return j0(*x); }
