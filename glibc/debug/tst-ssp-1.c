/* Verify that __stack_chk_fail won't segfault.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* Based on gcc.dg/ssp-1.c from GCC testsuite.  */

#include <signal.h>

static void
__attribute__ ((noinline, noclone))
test (char *foo)
{
  int i;

  /* smash stack */
  for (i = 0; i <= 400; i++)
    foo[i] = 42;
}

static int
do_test (void)
{
  char foo[30];

  test (foo);

  return 1; /* fail */
}

#define EXPECTED_SIGNAL SIGABRT
#include <support/test-driver.c>
