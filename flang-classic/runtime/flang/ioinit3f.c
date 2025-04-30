/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	ioinit3f.c - Implements LIB3F ioinit subprogram.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#define IS_TRUE(x) ((x)&0x1)

struct {
  short ieof;
  short ictl;
  short ibzr;
} ioiflg_ = {0};

void ENT3F(IOINIT, ioinit)(int *cctl, int *bzro, int *apnd, DCHAR(prefix),
                           int *vrbose DCLEN(prefix))
{
  /* stub for now */
}
