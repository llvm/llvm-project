/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	rand3f.c - Implements LIB3F rand subprogram.  */

#include "ent3f.h"

/* drand48 is not currently available on win64 */
#if defined(_WIN64)

#include <stdlib.h>

float ENT3F(RAND1, rand1)()
{
  float scale, base, fine;

  scale = RAND_MAX + 1.0;
  base = rand() / scale;
  fine = rand() / scale;
  return base + fine / scale;
}

float ENT3F(RAND2, rand2)(int *flag)
{
  float scale, base, fine;

  if (*flag)
    srand(*flag);
  scale = RAND_MAX + 1.0;
  base = rand() / scale;
  fine = rand() / scale;
  return base + fine / scale;
}
#else

extern double drand48();

double ENT3F(RAND, rand)(void) { return drand48(); }

#endif
