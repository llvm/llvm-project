/* Test local and global dynamic models for GNU2 TLS.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>

extern int * get_gd (void);
extern void set_gd (int);
extern int test_gd (int);
extern int * get_ld (void);
extern void set_ld (int);
extern int test_ld (int);

__thread int gd = 1;

static int
do_test (void)
{
  int *p;

  p = get_gd ();
  set_gd (3);
  if (*p != 3 || !test_gd (3))
    abort ();

  p = get_ld ();
  set_ld (4);
  if (*p != 4 || !test_ld (4))
    abort ();

  printf ("PASS\n");

  return 0;
}

#include <support/test-driver.c>
