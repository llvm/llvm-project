/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	derf3f.c - Implements LIB3F derf subprogram.  */

#include "ent3f.h"

extern double erf(double);

double ENT3F(DERF, derf)(double *x) { return erf(*x); }
