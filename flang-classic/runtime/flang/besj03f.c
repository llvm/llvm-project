/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	besj03f.c - Implements LIB3F besj0 subprogram.  */

#include "ent3f.h"

#if defined(_WIN64)
#define j0 _j0
#endif

extern double j0(double);

float ENT3F(BESJ0, besj0)(float *x) { return (float)j0(*x); }
