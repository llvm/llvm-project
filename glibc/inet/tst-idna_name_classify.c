/* Test IDNA name classification.
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

#include <inet/net-internal.h>
#include <locale.h>
#include <stdio.h>
#include <support/check.h>

static void
locale_insensitive_tests (void)
{
  TEST_COMPARE (__idna_name_classify (""), idna_name_ascii);
  TEST_COMPARE (__idna_name_classify ("abc"), idna_name_ascii);
  TEST_COMPARE (__idna_name_classify (".."), idna_name_ascii);
  TEST_COMPARE (__idna_name_classify ("\001abc\177"), idna_name_ascii);
  TEST_COMPARE (__idna_name_classify ("\\065bc"), idna_name_ascii);
}

static int
do_test (void)
{
  puts ("info: C locale tests");
  locale_insensitive_tests ();
  TEST_COMPARE (__idna_name_classify ("abc\200def"),
                idna_name_encoding_error);
  TEST_COMPARE (__idna_name_classify ("abc\200\\def"),
                idna_name_encoding_error);
  TEST_COMPARE (__idna_name_classify ("abc\377def"),
                idna_name_encoding_error);

  puts ("info: en_US.ISO-8859-1 locale tests");
  if (setlocale (LC_CTYPE, "en_US.ISO-8859-1") == 0)
    FAIL_EXIT1 ("setlocale for en_US.ISO-8859-1: %m\n");
  locale_insensitive_tests ();
  TEST_COMPARE (__idna_name_classify ("abc\200def"), idna_name_nonascii);
  TEST_COMPARE (__idna_name_classify ("abc\377def"), idna_name_nonascii);
  TEST_COMPARE (__idna_name_classify ("abc\\\200def"),
                idna_name_nonascii_backslash);
  TEST_COMPARE (__idna_name_classify ("abc\200\\def"),
                idna_name_nonascii_backslash);

  puts ("info: en_US.UTF-8 locale tests");
  if (setlocale (LC_CTYPE, "en_US.UTF-8") == 0)
    FAIL_EXIT1 ("setlocale for en_US.UTF-8: %m\n");
  locale_insensitive_tests ();
  TEST_COMPARE (__idna_name_classify ("abc\xc3\x9f""def"), idna_name_nonascii);
  TEST_COMPARE (__idna_name_classify ("abc\\\xc3\x9f""def"),
                idna_name_nonascii_backslash);
  TEST_COMPARE (__idna_name_classify ("abc\xc3\x9f\\def"),
                idna_name_nonascii_backslash);
  TEST_COMPARE (__idna_name_classify ("abc\200def"), idna_name_encoding_error);
  TEST_COMPARE (__idna_name_classify ("abc\xc3""def"), idna_name_encoding_error);
  TEST_COMPARE (__idna_name_classify ("abc\xc3"), idna_name_encoding_error);

  return 0;
}

#include <support/test-driver.c>
