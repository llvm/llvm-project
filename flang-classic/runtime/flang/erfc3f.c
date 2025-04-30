/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	erfc3f.c - Implements LIB3F erfc subprogram.  */

#include "ent3f.h"

extern double erfc(double);

float ENT3F(ERFC, erfc)(float *x) { return (float)erfc(*x); }
