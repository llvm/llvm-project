/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "ent3f.h"

static int
_isinff(volatile float x)
{
  union {
    float x;
    struct {
      unsigned int m : 23;
      unsigned int e : 8;
      unsigned int s : 1;
    } f;
  } u;

  u.x = x;
  return (u.f.e == 255 && u.f.m == 0);
}

int ENT3F(ISINFF, isinff)(float *x)
{
  if (_isinff(*x))
    return -1; /* .true. */
  return 0;
}
