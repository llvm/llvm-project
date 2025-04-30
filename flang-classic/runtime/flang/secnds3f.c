/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	secnds3f.c - Implements LIB3F secnds subprogram.  */

#include "ent3f.h"

#include "enames.h"

extern float Ftn_secnds(float);

float ENT3F(SECNDS, secnds)(float *x) { return Ftn_secnds(*x); }
