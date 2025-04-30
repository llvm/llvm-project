/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <stdbool.h>
#include <stdio.h>
#include <tst-stack-align.h>
#include <unistd.h>

static int res, *fdp;
static bool *test_destructorsp;

static void __attribute__((constructor))
con (void)
{
  res = TEST_STACK_ALIGN () ? -1 : 1;
}

void
in_dso (int *result, bool *test_destructors, int *fd)
{
  if (!res)
    {
      puts ("constructor has not been run");
      *result = 1;
    }
  else if (res != 1)
    {
      puts ("constructor has been run without sufficient alignment");
      *result = 1;
    }

  test_destructorsp = test_destructors;
  fdp = fd;
}

static void __attribute__((destructor))
des (void)
{
  if (!test_destructorsp || !*test_destructorsp)
    return;

  char c = TEST_STACK_ALIGN () ? 'D' : 'C';
  write (*fdp, &c, 1);
}
