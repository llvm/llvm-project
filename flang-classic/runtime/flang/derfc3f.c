/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	derfc3f.c - Implements LIB3F derfc subprogram.  */

#include "ent3f.h"

extern double erfc(double);

double ENT3F(DERFC, derfc)(double *x) { return erfc(*x); }
