/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	srand3f.c - Implements LIB3F srand subprogram.  */

#include <stdlib.h>
#include "ent3f.h"

/* srand48 is not currently available on win64 */
#if defined(_WIN64)

void ENT3F(SRAND1, srand1)(int *iseed) { srand(*iseed); }

void ENT3F(SRAND2, srand2)(float *rseed)
{
  int iseed;
  iseed = (int)(*rseed);
  srand(iseed);
}

#else

void ENT3F(SRAND, srand)(int *iseed) { srand48(*iseed); }

#endif
