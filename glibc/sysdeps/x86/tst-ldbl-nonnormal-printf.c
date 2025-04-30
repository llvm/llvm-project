/* Test printf with x86-specific non-normal long double value.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.

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
#include <string.h>
#include <support/check.h>

/* Fill the stack with non-zero values.  This makes a crash in
   snprintf more likely.  */
static void __attribute__ ((noinline, noclone))
fill_stack (void)
{
  char buffer[65536];
  memset (buffer, 0xc0, sizeof (buffer));
  asm ("" ::: "memory");
}

static int
do_test (void)
{
  fill_stack ();

  long double value;
  memcpy (&value, "\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04", 10);

  char buf[30];
  int ret = snprintf (buf, sizeof (buf), "%Lg", value);
  TEST_COMPARE (ret, strlen (buf));
  TEST_COMPARE_STRING (buf, "nan");
  return 0;
}

#include <support/test-driver.c>
