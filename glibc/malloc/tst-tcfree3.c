/* Test that malloc tcache catches double free.
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

#include <malloc.h>
#include <string.h>

/* Prevent GCC from optimizing away any malloc/free pairs.  */
#pragma GCC optimize ("O0")

static int
do_test (void)
{
  /* Do two allocation of any size that fit in tcache, and one that
     doesn't.  */
  int ** volatile a = malloc (32);
  int ** volatile b = malloc (32);
  /* This is just under the mmap threshold.  */
  int ** volatile c = malloc (127 * 1024);

  /* The invalid "tcache bucket" we might dereference will likely end
     up somewhere within this memory block, so make all the accidental
     "next" pointers cause segfaults.  BZ #23907.  */
  memset (c, 0xff, 127 * 1024);

  free (a); // puts in tcache

  /* A is now free and contains the key we use to detect in-tcache.
     Copy the key to the other chunks.  */
  memcpy (b, a, 32);
  memcpy (c, a, 32);

  /* This free tests the "are we in the tcache already" loop with a
     VALID bin but "coincidental" matching key.  */
  free (b); // should NOT abort
  /* This free tests the "is it a valid tcache bin" test.  */
  free (c); // should NOT abort

  return 0;
}

#include <support/test-driver.c>
