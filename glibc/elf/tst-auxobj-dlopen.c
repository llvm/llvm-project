/* Test for BZ#16272, dlopen'ing an auxiliary filter object.
   Ensure that symbols from the resolve correctly.

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
#include <support/xdlfcn.h>

static int do_test (void)
{
  void *lib = xdlopen ("tst-filterobj-aux.so", RTLD_LAZY);
  char *(*fn)(void) = xdlsym (lib, "get_text");
  const char* text = fn ();

  printf ("%s\n", text);

  /* Verify the text matches what we expect from the filtee */
  TEST_COMPARE_STRING (text, "Hello from filtee (PASS)");

  fn = xdlsym (lib, "get_text2");
  text = fn ();

  printf ("%s\n", text);

  /* Verify the text matches what we expect from the auxiliary object */
  TEST_COMPARE_STRING (text, "Hello from auxiliary filter object (PASS)");

  return 0;
}

#include <support/test-driver.c>
