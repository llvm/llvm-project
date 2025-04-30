/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	erf3f.c - Implements LIB3F erf subprogram.  */

#include "ent3f.h"

extern double erf(double);

float ENT3F(ERF, erf)(float *x) { return (float)erf(*x); }
