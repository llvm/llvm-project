/* Tests for strfromf, strfromd, strfroml functions.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include "tst-strfrom.h"

static const struct test tests[] = {
  TEST ("12.345000", "%f", 50, 9, 12.345),
  TEST ("9.999", "%.3f", 50, 5, 9.999),
  TEST ("0.125000", "%f", 50, 8, .125),
  TEST ("0.000000", "%f", 50, 8, .0),
  TEST ("0", "%g", 50, 1, .0),
  TEST ("9.900000", "%f", 50, 8, 9.9),
  TEST ("9.1", "%.5f", 4, 7, 9.123456),
  TEST ("9.91235", "%g", 50, 7, 9.91234567812345678),
  TEST ("79.8765", "%G", 50, 7, 79.8765432111),
  TEST ("79.9", "%.3g", 50, 4, 79.8765432111),
  TEST ("1.000000e+38", "%e", 50, 12, 1e+38),
  TEST ("1.000000e+38", "%e", 50, 12, 1e38),
  TEST ("-1.000000e-37", "%e", 50, 13, -1e-37),
  TEST ("1.000000e-37", "%e", 50, 12, 0.00000001e-29),
  TEST ("1.000000e-37", "%e", 50, 12, 1.000000e-37),
  TEST ("5.900000e-16", "%e", 50, 12, 5.9e-16),
  TEST ("1.234500e+20", "%e", 50, 12, 12.345e19),
  TEST ("1.000000e+05", "%e", 50, 12, 1e5),
  TEST ("-NAN", "%G", 50, 4, -NAN_),
  TEST ("-inf", "%g", 50, 4, -INF),
  TEST ("inf", "%g", 50, 3, INF)
};
/* Tests with buffer size small.  */
static const struct test stest[] = {
  TEST ("1234", "%g", 5, 7, 12345.345),
  TEST ("0.12", "%f", 5, 8, .125),
  TEST ("9.99", "%.3f", 5, 5, 9.999),
  TEST ("100", "%g", 5, 3, 1e2)
};
/* Hexadecimal tests.  */
static const struct htests htest[] = {
  HTEST ("%a", { "0x1.ffp+6", "0x3.fep+5", "0x7.fcp+4", "0xf.f8p+3" },
	0x1.ffp+6),
  HTEST ("%a", { "0x1.88p+4", "0x3.1p+3", "0x6.2p+2", "0xc.4p+1" },
	0x1.88p+4),
  HTEST ("%A", { "-0X1.88P+5", "-0X3.1P+4", "-0X6.2P+3", "-0XC.4P+2" },
	-0x1.88p+5),
  HTEST ("%a", { "0x1.44p+10", "0x2.88p+9", "0x5.1p+8", "0xa.2p+7" },
	0x1.44p+10),
  HTEST ("%a", { "0x1p-10", "0x2p-11", "0x4p-12", "0x8p-13" },
	0x0.0040p+0),
  HTEST ("%a", { "0x1.4p+3", "0x2.8p+2", "0x5p+1", "0xap+0" },
	10.0)
};

GEN_TEST_STRTOD_FOREACH (TEST_STRFROM)

static int
test_locale (const char *locale)
{
  printf ("Testing in locale: %s\n", locale);
  if (setlocale (LC_ALL, locale) == NULL)
    {
      printf ("Cannot set locale %s\n", locale);
    }
  return STRTOD_TEST_FOREACH (test_);
}

static int
do_test (void)
{
  int result = 0;
  result += test_locale ("C");
  result += test_locale ("en_US.ISO-8859-1");
  result += test_locale ("en_US.UTF-8");
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
