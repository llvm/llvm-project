/* Test that symbols from auxiliary filter objects are resolved to the
   filtee.

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
#include <support/check.h>
#include "tst-filterobj-filtee.h"

static int do_test (void)
{
  const char* text = get_text ();
  printf ("%s\n", text);

  /* Verify the text matches what we expect from the filtee */
  TEST_COMPARE_STRING (text, "Hello from filtee (PASS)");

  text = get_text2 ();
  printf ("%s\n", text);

  /* Verify the text matches what we expect from the auxiliary object */
  TEST_COMPARE_STRING (text, "Hello from auxiliary filter object (PASS)");

  return 0;
}

#include <support/test-driver.c>
