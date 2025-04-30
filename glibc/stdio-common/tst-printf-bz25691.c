/* Test for memory leak with large width (BZ#25691).
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
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <stdint.h>
#include <locale.h>

#include <mcheck.h>
#include <support/check.h>
#include <support/support.h>

static int
do_test (void)
{
  mtrace ();

  /* For 's' conversion specifier with 'l' modifier the array must be
     converted to multibyte characters up to the precision specific
     value.  */
  {
    /* The input size value is to force a heap allocation on temporary
       buffer (in the old implementation).  */
    const size_t winputsize = 64 * 1024 + 1;
    wchar_t *winput = xmalloc (winputsize * sizeof (wchar_t));
    wmemset (winput, L'a', winputsize - 1);
    winput[winputsize - 1] = L'\0';

    char result[9];
    const char expected[] = "aaaaaaaa";
    int ret;

    ret = snprintf (result, sizeof (result), "%.65537ls", winput);
    TEST_COMPARE (ret, winputsize - 1);
    TEST_COMPARE_BLOB (result, sizeof (result), expected, sizeof (expected));

    ret = snprintf (result, sizeof (result), "%ls", winput);
    TEST_COMPARE (ret, winputsize - 1);
    TEST_COMPARE_BLOB (result, sizeof (result), expected, sizeof (expected));

    free (winput);
  }

  /* For 's' converstion specifier the array is interpreted as a multibyte
     character sequence and converted to wide characters up to the precision
     specific value.  */
  {
    /* The input size value is to force a heap allocation on temporary
       buffer (in the old implementation).  */
    const size_t mbssize = 32 * 1024;
    char *mbs = xmalloc (mbssize);
    memset (mbs, 'a', mbssize - 1);
    mbs[mbssize - 1] = '\0';

    const size_t expectedsize = 32 * 1024;
    wchar_t *expected = xmalloc (expectedsize * sizeof (wchar_t));
    wmemset (expected, L'a', expectedsize - 1);
    expected[expectedsize-1] = L'\0';

    const size_t resultsize = mbssize * sizeof (wchar_t);
    wchar_t *result = xmalloc (resultsize);
    int ret;

    ret = swprintf (result, resultsize, L"%.65537s", mbs);
    TEST_COMPARE (ret, mbssize - 1);
    TEST_COMPARE_BLOB (result, (ret + 1) * sizeof (wchar_t),
		       expected, expectedsize * sizeof (wchar_t));

    ret = swprintf (result, resultsize, L"%1$.65537s", mbs);
    TEST_COMPARE (ret, mbssize - 1);
    TEST_COMPARE_BLOB (result, (ret + 1) * sizeof (wchar_t),
		       expected, expectedsize * sizeof (wchar_t));

    /* Same test, but with an invalid multibyte sequence.  */
    mbs[mbssize - 2] = 0xff;

    ret = swprintf (result, resultsize, L"%.65537s", mbs);
    TEST_COMPARE (ret, -1);

    ret = swprintf (result, resultsize, L"%1$.65537s", mbs);
    TEST_COMPARE (ret, -1);

    free (mbs);
    free (result);
    free (expected);
  }

  return 0;
}

#include <support/test-driver.c>
