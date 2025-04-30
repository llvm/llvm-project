/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	besj13f.c - Implements LIB3F besj1 subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define j1 _j1
#endif

extern double j1(double);

float ENT3F(BESJ1, besj1)(float *x) { return (float)j1(*x); }
