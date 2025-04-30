/* Test basic mbstowcs including wstring == NULL (Bug 25219).
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

#include <stdlib.h>
#include <string.h>
#include <support/check.h>

static int
do_test (void)
{
  char string[] = { '1', '2', '3' , '4', '5', '\0' };
  size_t len = strlen (string);
  wchar_t wstring[] = { L'1', L'2', L'3', L'4', L'5', L'\0' };
#define NUM_WCHAR 6
  wchar_t wout[NUM_WCHAR];
  size_t result;

  /* The input ASCII string in the C/POSIX locale must convert
     to the matching WSTRING.  */
  result = mbstowcs (wout, string, NUM_WCHAR);
  TEST_VERIFY (result == (NUM_WCHAR - 1));
  TEST_COMPARE_BLOB (wstring, sizeof (wchar_t) * (NUM_WCHAR - 1),
		     wout, sizeof (wchar_t) * result);

  /* The input ASCII string in the C/POSIX locale must be the
     same length when using mbstowcs to compute the length of
     the string required in the conversion.  Using mbstowcs
     in this way is an XSI extension to POSIX.  */
  result = mbstowcs (NULL, string, len);
  TEST_VERIFY (result == len);

  return 0;
}

#include <support/test-driver.c>
