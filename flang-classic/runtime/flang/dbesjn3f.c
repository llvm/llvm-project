/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	dbesjn3f.c - Implements LIB3F dbesjn subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define jn _jn
#endif

extern double jn(int, double);

double ENT3F(DBESJN, dbesjn)(int *n, double *x) { return jn(*n, *x); }
