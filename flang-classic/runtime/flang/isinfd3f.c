/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "ent3f.h"

static int
_isinfd(volatile double x)
{
  union {
    double x;
    struct {
      unsigned int ml;
      unsigned int mh : 20;
      unsigned int e : 11;
      unsigned int s : 1;
    } f;
  } u;

  u.x = x;
  return (u.f.e == 2047 && (u.f.ml == 0 && u.f.mh == 0));
}

int ENT3F(ISINFD, isinfd)(double *x)
{
  if (_isinfd(*x))
    return -1; /* .true. */
  return 0;
}
