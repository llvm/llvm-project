/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	drandm3f.c - Implements LIB3F drandm subprogram.  */

#include <stdlib.h>
#include "ent3f.h"

/* drand48, srand48 are not currently available on win64 */
#if defined(_WIN64)

double ENT3F(DRANDM, drandm)(int *flag)
{
  double scale, base, fine;

  if (*flag)
    srand(*flag);
  scale = RAND_MAX + 1.0;
  base = rand() / scale;
  fine = rand() / scale;
  return base + fine / scale;
}

double ENT3F(DRAND, drand)(int *flag)
{
  return drandm_(flag);
}

#else

double ENT3F(DRANDM, drandm)(int *flag)
{
  if (*flag)
    srand48(*flag);
  return drand48();
}

#endif
