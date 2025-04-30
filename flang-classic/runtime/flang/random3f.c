/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	random3f.c - Implements LIB3F random subprogram.  */

#include <stdlib.h>
#include "ent3f.h"

/* drand48, srand48 are not currently available on win64 */
#if defined(_WIN64)

float ENT3F(RANDOM, random)(int *flag)
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

float ENT3F(RANDOM, random)(int *flag)
{
  if (*flag)
    srand48(*flag);
  return (float)drand48();
}

#endif
