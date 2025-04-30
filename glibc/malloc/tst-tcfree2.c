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

#include <errno.h>
#include <error.h>
#include <limits.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/signal.h>

static int
do_test (void)
{
#define COUNT 20
  char * volatile ptrs[COUNT];
  int i;

  /* Allocate enough small chunks so that when we free them all, the tcache
     is full, and the first one we freed is at the end of its linked list.  */
  for (i = 0; i < COUNT; i++)
    ptrs[i] = malloc (20);
  for (i = 0; i < COUNT; i++)
    free (ptrs[i]);
  free (ptrs[0]);

  printf("FAIL: tcache double free\n");
  return 1;
}

#define TEST_FUNCTION do_test
#define EXPECTED_SIGNAL SIGABRT
#include <support/test-driver.c>
