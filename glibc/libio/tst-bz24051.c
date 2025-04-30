/* Test that assigning to stdout redirects puts, putchar, etc (BZ#24051)
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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


/* Prevent putchar -> _IO_putc inline expansion.  */
#define __NO_INLINE__
#pragma GCC optimize("O0")

#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include <array_length.h>
#include <support/check.h>
#include <support/temp_file.h>
#include <support/test-driver.h>

#undef putchar
#undef putwchar

static int
do_test_narrow (void)
{
  char buf[100];
  int fd = create_temp_file ("tst-bz24051", NULL);
  stdout = fdopen (fd, "w+");
  TEST_VERIFY_EXIT (stdout != NULL);

  printf ("ab%s", "cd");
  putchar ('e');
  putchar_unlocked ('f');
  puts ("ghi");

  rewind (stdout);
  TEST_VERIFY_EXIT (fgets (buf, sizeof (buf), stdout) != NULL);
  TEST_VERIFY (strcmp (buf, "abcdefghi\n") == 0);

  return 0;
}

static int
do_test_wide (void)
{
  wchar_t buf[100];
  int fd = create_temp_file ("tst-bz24051w", NULL);
  stdout = fdopen (fd, "w+");
  TEST_VERIFY_EXIT (stdout != NULL);

  wprintf (L"ab%ls", L"cd");
  putwchar (L'e');
  putwchar_unlocked (L'f');

  rewind (stdout);
  TEST_VERIFY_EXIT (fgetws (buf, array_length (buf), stdout) != NULL);
  TEST_VERIFY (wcscmp (buf, L"abcdef") == 0);

  return 0;
}

static int
do_test (void)
{
  return do_test_narrow () + do_test_wide ();
}

#include <support/test-driver.c>
