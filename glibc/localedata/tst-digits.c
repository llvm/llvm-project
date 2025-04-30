/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 2000.

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

#include <ctype.h>
#include <langinfo.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>
#include <wctype.h>
#include <sys/types.h>


#define ZERO  "\xe2\x82\x80"
#define ONE   "\xe2\x82\x81"
#define TWO   "\xe2\x82\x82"
#define THREE "\xe2\x82\x83"
#define FOUR  "\xe2\x82\x84"
#define FIVE  "\xe2\x82\x85"
#define SIX   "\xe2\x82\x86"
#define SEVEN "\xe2\x82\x87"
#define EIGHT "\xe2\x82\x88"
#define NINE  "\xe2\x82\x89"

static struct printf_int_test
{
  int n;
  const char *format;
  const char *expected;
} printf_int_tests[] =
{
  {       0, "%I'10d", "       " ZERO },
  {       1, "%I'10d", "       " ONE },
  {       2, "%I'10d", "       " TWO },
  {       3, "%I'10d", "       " THREE },
  {       4, "%I'10d", "       " FOUR },
  {       5, "%I'10d", "       " FIVE },
  {       6, "%I'10d", "       " SIX },
  {       7, "%I'10d", "       " SEVEN },
  {       8, "%I'10d", "       " EIGHT },
  {       9, "%I'10d", "       " NINE },
  {      11, "%I'10d", "    " ONE ONE },
  {      12, "%I'10d", "    " ONE TWO },
  {     123, "%I10d",  " " ONE TWO THREE },
  {     123, "%I'10d", " " ONE TWO THREE },
  {    1234, "%I10d",  ONE TWO THREE FOUR },
  {    1234, "%I'10d", ONE "," TWO THREE FOUR },
  {   12345, "%I'10d", ONE TWO "," THREE FOUR FIVE },
  {  123456, "%I'10d", ONE TWO THREE "," FOUR FIVE SIX },
  { 1234567, "%I'10d", ONE "," TWO THREE FOUR "," FIVE SIX SEVEN }
};
#define nprintf_int_tests \
  (sizeof (printf_int_tests) / sizeof (printf_int_tests[0]))

#define WZERO  L"\x2080"
#define WONE   L"\x2081"
#define WTWO   L"\x2082"
#define WTHREE L"\x2083"
#define WFOUR  L"\x2084"
#define WFIVE  L"\x2085"
#define WSIX   L"\x2086"
#define WSEVEN L"\x2087"
#define WEIGHT L"\x2088"
#define WNINE  L"\x2089"

static struct wprintf_int_test
{
  int n;
  const wchar_t *format;
  const wchar_t *expected;
} wprintf_int_tests[] =
{
  {       0, L"%I'10d", L"         " WZERO },
  {       1, L"%I'10d", L"         " WONE },
  {       2, L"%I'10d", L"         " WTWO },
  {       3, L"%I'10d", L"         " WTHREE },
  {       4, L"%I'10d", L"         " WFOUR },
  {       5, L"%I'10d", L"         " WFIVE },
  {       6, L"%I'10d", L"         " WSIX },
  {       7, L"%I'10d", L"         " WSEVEN },
  {       8, L"%I'10d", L"         " WEIGHT },
  {       9, L"%I'10d", L"         " WNINE },
  {      11, L"%I'10d", L"        " WONE WONE },
  {      12, L"%I'10d", L"        " WONE WTWO },
  {     123, L"%I10d",  L"       " WONE WTWO WTHREE },
  {     123, L"%I'10d", L"       " WONE WTWO WTHREE },
  {    1234, L"%I10d",  L"      " WONE WTWO WTHREE WFOUR },
  {    1234, L"%I'10d", L"     " WONE L"," WTWO WTHREE WFOUR },
  {   12345, L"%I'10d", L"    " WONE WTWO L"," WTHREE WFOUR WFIVE },
  {  123456, L"%I'10d", L"   " WONE WTWO WTHREE L"," WFOUR WFIVE WSIX },
  { 1234567, L"%I'10d", L" " WONE L"," WTWO WTHREE WFOUR L"," WFIVE WSIX WSEVEN }
};
#define nwprintf_int_tests \
  (sizeof (wprintf_int_tests) / sizeof (wprintf_int_tests[0]))


static int
do_test (void)
{
  int cnt;
  int failures;
  int status;

  if (setlocale (LC_ALL, "test7") == NULL)
    {
      puts ("cannot set locale `test7'");
      exit (1);
    }
  printf ("CODESET = \"%s\"\n", nl_langinfo (CODESET));

  /* First: printf tests.  */
  failures = 0;
  for (cnt = 0; cnt < (int) nprintf_int_tests; ++cnt)
    {
      char buf[100];
      ssize_t n;

      n = snprintf (buf, sizeof buf, printf_int_tests[cnt].format,
		    printf_int_tests[cnt].n);

      printf ("%3d: got \"%s\", expected \"%s\"",
	      cnt, buf, printf_int_tests[cnt].expected);

      if (n != (ssize_t) strlen (printf_int_tests[cnt].expected)
	  || strcmp (buf, printf_int_tests[cnt].expected) != 0)
	{
	  puts ("  -> FAILED");
	  ++failures;
	}
      else
	puts ("  -> OK");
    }

  printf ("%d failures in printf tests\n", failures);
  status = failures != 0;

  /* wprintf tests.  */
  failures = 0;
  for (cnt = 0; cnt < (int) nwprintf_int_tests; ++cnt)
    {
      wchar_t buf[100];
      ssize_t n;

      n = swprintf (buf, sizeof buf / sizeof (buf[0]),
		    wprintf_int_tests[cnt].format,
		    wprintf_int_tests[cnt].n);

      printf ("%3d: got \"%ls\", expected \"%ls\"",
	      cnt, buf, wprintf_int_tests[cnt].expected);

      if (n != (ssize_t) wcslen (wprintf_int_tests[cnt].expected)
	  || wcscmp (buf, wprintf_int_tests[cnt].expected) != 0)
	{
	  puts ("  -> FAILED");
	  ++failures;
	}
      else
	puts ("  -> OK");
    }

  printf ("%d failures in wprintf tests\n", failures);
  status = failures != 0;

  /* ctype tests.  This makes sure that the multibyte character digit
     representations are not handle in this table.  */
  failures = 0;
  for (cnt = 0; cnt < 256; ++cnt)
    if (cnt >= '0' && cnt <= '9')
      {
	if (! isdigit (cnt))
	  {
	    printf ("isdigit ('%c') == 0\n", cnt);
	    ++failures;
	  }
      }
    else
      {
	if (isdigit (cnt))
	  {
	    printf ("isdigit (%d) != 0\n", cnt);
	    ++failures;
	  }
      }

  printf ("%d failures in ctype tests\n", failures);
  status = failures != 0;

  /* wctype tests.  This makes sure the second set of digits is also
     recorded.  */
  failures = 0;
  for (cnt = 0; cnt < 256; ++cnt)
    if (cnt >= '0' && cnt <= '9')
      {
	if (! iswdigit (cnt))
	  {
	    printf ("iswdigit (L'%c') == 0\n", cnt);
	    ++failures;
	  }
      }
    else
      {
	if (iswdigit (cnt))
	  {
	    printf ("iswdigit (%d) != 0\n", cnt);
	    ++failures;
	  }
      }

  for (cnt = 0x2070; cnt < 0x2090; ++cnt)
    if (cnt >= 0x2080 && cnt <= 0x2089)
      {
	if (! iswdigit (cnt))
	  {
	    printf ("iswdigit (U%04X) == 0\n", cnt);
	    ++failures;
	  }
      }
    else
      {
	if (iswdigit (cnt))
	  {
	    printf ("iswdigit (U%04X) != 0\n", cnt);
	    ++failures;
	  }
      }

  printf ("%d failures in wctype tests\n", failures);
  status = failures != 0;

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
