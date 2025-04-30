/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	dsecnds3f.c - Implements LIB3F dsecnds function.
 *
 * Returns the number of real time seconds since midnight minus the supplied
 */

#include "ent3f.h"

#include "enames.h"

extern double Ftn_dsecnds(double);

double ENT3F(DSECNDS, dsecnds)(double *x) { return Ftn_dsecnds(*x); }
