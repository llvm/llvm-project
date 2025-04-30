/* Test code declaring narrowing functions does not conflict with user macros.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* The code generating declarations of narrowing functions involves
   concatenations of fragments of function names that are not
   themselves reserved; thus, it needs to be arranged so that those
   fragments are not subject to macro expansion.  Verify that
   inclusion of <math.h> compiles with such fragments defined as
   macros.  */

#define f test macro
#define d test macro
#define l test macro
#define f16 test macro
#define f32 test macro
#define f64 test macro
#define f128 test macro
#define f32x test macro
#define f64x test macro
#define f128x test macro
#define add test macro
#define sub test macro
#define mul test macro
#define div test macro
#define dadd test macro
#define dsub test macro
#define dmul test macro
#define ddiv test macro
#define dsqrt test macro
#define dfma test macro

#include <math.h>

static int
do_test (void)
{
  /* This is a compilation test.  */
  return 0;
}

#include <support/test-driver.c>
