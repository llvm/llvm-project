/* Ensure that malloc_usable_size returns the request size with
   MALLOC_CHECK_ exported to a positive value.

   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <stdio.h>

static int
do_test (void)
{
  size_t usable_size;
  void *p = malloc (7);
  if (!p)
    {
      printf ("memory allocation failed\n");
      return 1;
    }

  usable_size = malloc_usable_size (p);
  if (usable_size != 7)
    {
      printf ("malloc_usable_size: expected 7 but got %zu\n", usable_size);
      return 1;
    }

  memset (p, 0, usable_size);
  free (p);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
