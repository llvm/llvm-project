/* Test for correct rounding of printf floating-point output.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <array_length.h>
#include <fenv.h>
#include <stdio.h>
#include <string.h>

struct dec_test {
  double d;
  const char *fmt;
  const char *rd, *rn, *rz, *ru;
};

static const struct dec_test dec_tests[] = {
  { 1.5, "%.0f", "1", "2", "1", "2" },
  { -1.5, "%.0f", "-2", "-2", "-1", "-1" },
  { 2.5, "%.0f", "2", "2", "2", "3" },
  { -2.5, "%.0f", "-3", "-2", "-2", "-2" },
  { 1.4999, "%.0f", "1", "1", "1", "2" },
  { -1.4999, "%.0f", "-2", "-1", "-1", "-1" },
  { 1.5001, "%.0f", "1", "2", "1", "2" },
  { -1.5001, "%.0f", "-2", "-2", "-1", "-1" },
  { 2.4999, "%.0f", "2", "2", "2", "3" },
  { -2.4999, "%.0f", "-3", "-2", "-2", "-2" },
  { 2.5001, "%.0f", "2", "3", "2", "3" },
  { -2.5001, "%.0f", "-3", "-3", "-2", "-2" },
  { 1.0 / 3.0, "%f", "0.333333", "0.333333", "0.333333", "0.333334" },
  { -1.0 / 3.0, "%f", "-0.333334", "-0.333333", "-0.333333", "-0.333333" },
  { 0.2500001, "%.2e", "2.50e-01", "2.50e-01", "2.50e-01", "2.51e-01" },
  { -0.2500001, "%.2e", "-2.51e-01", "-2.50e-01", "-2.50e-01", "-2.50e-01" },
  { 1000001.0, "%.1e", "1.0e+06", "1.0e+06", "1.0e+06", "1.1e+06" },
  { -1000001.0, "%.1e", "-1.1e+06", "-1.0e+06", "-1.0e+06", "-1.0e+06" },
};

static int
test_dec_in_one_mode (double d, const char *fmt, const char *expected,
		      const char *mode_name)
{
  char buf[100];
  int ret = snprintf (buf, sizeof buf, fmt, d);
  if (ret <= 0 || ret >= (int) sizeof buf)
    {
      printf ("snprintf for %a returned %d\n", d, ret);
      return 1;
    }
  if (strcmp (buf, expected) == 0)
    return 0;
  else
    {
      printf ("snprintf (\"%s\", %a) returned \"%s\" not \"%s\" (%s)\n",
	      fmt, d, buf, expected, mode_name);
      return 1;
    }
}

struct hex_test
{
  double d;
  const char *fmt;
  const char *rd[4], *rn[4], *rz[4], *ru[4];
};

static const struct hex_test hex_tests[] =
  {
    {
      0x1.fffffp+4, "%.1a",
      { "0x1.fp+4", "0x3.fp+3", "0x7.fp+2", "0xf.fp+1" },
      { "0x2.0p+4", "0x4.0p+3", "0x8.0p+2", "0x1.0p+5" },
      { "0x1.fp+4", "0x3.fp+3", "0x7.fp+2", "0xf.fp+1" },
      { "0x2.0p+4", "0x4.0p+3", "0x8.0p+2", "0x1.0p+5" }
    },
    {
      -0x1.fffffp+4, "%.1a",
      { "-0x2.0p+4", "-0x4.0p+3", "-0x8.0p+2", "-0x1.0p+5" },
      { "-0x2.0p+4", "-0x4.0p+3", "-0x8.0p+2", "-0x1.0p+5" },
      { "-0x1.fp+4", "-0x3.fp+3", "-0x7.fp+2", "-0xf.fp+1" },
      { "-0x1.fp+4", "-0x3.fp+3", "-0x7.fp+2", "-0xf.fp+1" }
    },
    {
      0x1.88p+4, "%.1a",
      { "0x1.8p+4", "0x3.1p+3", "0x6.2p+2", "0xc.4p+1" },
      { "0x1.8p+4", "0x3.1p+3", "0x6.2p+2", "0xc.4p+1" },
      { "0x1.8p+4", "0x3.1p+3", "0x6.2p+2", "0xc.4p+1" },
      { "0x1.9p+4", "0x3.1p+3", "0x6.2p+2", "0xc.4p+1" }
    },
    {
      -0x1.88p+4, "%.1a",
      { "-0x1.9p+4", "-0x3.1p+3", "-0x6.2p+2", "-0xc.4p+1" },
      { "-0x1.8p+4", "-0x3.1p+3", "-0x6.2p+2", "-0xc.4p+1" },
      { "-0x1.8p+4", "-0x3.1p+3", "-0x6.2p+2", "-0xc.4p+1" },
      { "-0x1.8p+4", "-0x3.1p+3", "-0x6.2p+2", "-0xc.4p+1" }
    },
    {
      0x1.78p+4, "%.1a",
      { "0x1.7p+4", "0x2.fp+3", "0x5.ep+2", "0xb.cp+1" },
      { "0x1.8p+4", "0x2.fp+3", "0x5.ep+2", "0xb.cp+1" },
      { "0x1.7p+4", "0x2.fp+3", "0x5.ep+2", "0xb.cp+1" },
      { "0x1.8p+4", "0x2.fp+3", "0x5.ep+2", "0xb.cp+1" }
    },
    {
      -0x1.78p+4, "%.1a",
      { "-0x1.8p+4", "-0x2.fp+3", "-0x5.ep+2", "-0xb.cp+1" },
      { "-0x1.8p+4", "-0x2.fp+3", "-0x5.ep+2", "-0xb.cp+1" },
      { "-0x1.7p+4", "-0x2.fp+3", "-0x5.ep+2", "-0xb.cp+1" },
      { "-0x1.7p+4", "-0x2.fp+3", "-0x5.ep+2", "-0xb.cp+1" }
    },
    {
      64.0 / 3.0, "%.1a",
      { "0x1.5p+4", "0x2.ap+3", "0x5.5p+2", "0xa.ap+1" },
      { "0x1.5p+4", "0x2.bp+3", "0x5.5p+2", "0xa.bp+1" },
      { "0x1.5p+4", "0x2.ap+3", "0x5.5p+2", "0xa.ap+1" },
      { "0x1.6p+4", "0x2.bp+3", "0x5.6p+2", "0xa.bp+1" }
    },
    {
      -64.0 / 3.0, "%.1a",
      { "-0x1.6p+4", "-0x2.bp+3", "-0x5.6p+2", "-0xa.bp+1" },
      { "-0x1.5p+4", "-0x2.bp+3", "-0x5.5p+2", "-0xa.bp+1" },
      { "-0x1.5p+4", "-0x2.ap+3", "-0x5.5p+2", "-0xa.ap+1" },
      { "-0x1.5p+4", "-0x2.ap+3", "-0x5.5p+2", "-0xa.ap+1" }
    },
  };

static int
test_hex_in_one_mode (double d, const char *fmt, const char *const expected[4],
		      const char *mode_name)
{
  char buf[100];
  int ret = snprintf (buf, sizeof buf, fmt, d);
  if (ret <= 0 || ret >= (int) sizeof buf)
    {
      printf ("snprintf for %a returned %d\n", d, ret);
      return 1;
    }
  if (strcmp (buf, expected[0]) == 0
      || strcmp (buf, expected[1]) == 0
      || strcmp (buf, expected[2]) == 0
      || strcmp (buf, expected[3]) == 0)
    return 0;
  else
    {
      printf ("snprintf (\"%s\", %a) returned \"%s\" not "
	      "\"%s\" or \"%s\" or \"%s\" or \"%s\" (%s)\n",
	      fmt, d, buf, expected[0], expected[1], expected[2], expected[3],
	      mode_name);
      return 1;
    }
}

static int
do_test (void)
{
  int save_round_mode __attribute__ ((unused)) = fegetround ();
  int result = 0;

  for (size_t i = 0; i < array_length (dec_tests); i++)
    {
      result |= test_dec_in_one_mode (dec_tests[i].d, dec_tests[i].fmt,
				      dec_tests[i].rn, "default rounding mode");
#ifdef FE_DOWNWARD
      if (!fesetround (FE_DOWNWARD))
	{
	  result |= test_dec_in_one_mode (dec_tests[i].d, dec_tests[i].fmt,
					  dec_tests[i].rd, "FE_DOWNWARD");
	  fesetround (save_round_mode);
	}
#endif
#ifdef FE_TOWARDZERO
      if (!fesetround (FE_TOWARDZERO))
	{
	  result |= test_dec_in_one_mode (dec_tests[i].d, dec_tests[i].fmt,
					  dec_tests[i].rz, "FE_TOWARDZERO");
	  fesetround (save_round_mode);
	}
#endif
#ifdef FE_UPWARD
      if (!fesetround (FE_UPWARD))
	{
	  result |= test_dec_in_one_mode (dec_tests[i].d, dec_tests[i].fmt,
					  dec_tests[i].ru, "FE_UPWARD");
	  fesetround (save_round_mode);
	}
#endif
    }

  for (size_t i = 0; i < array_length (hex_tests); i++)
    {
      result |= test_hex_in_one_mode (hex_tests[i].d, hex_tests[i].fmt,
				      hex_tests[i].rn, "default rounding mode");
#ifdef FE_DOWNWARD
      if (!fesetround (FE_DOWNWARD))
	{
	  result |= test_hex_in_one_mode (hex_tests[i].d, hex_tests[i].fmt,
					  hex_tests[i].rd, "FE_DOWNWARD");
	  fesetround (save_round_mode);
	}
#endif
#ifdef FE_TOWARDZERO
      if (!fesetround (FE_TOWARDZERO))
	{
	  result |= test_hex_in_one_mode (hex_tests[i].d, hex_tests[i].fmt,
					  hex_tests[i].rz, "FE_TOWARDZERO");
	  fesetround (save_round_mode);
	}
#endif
#ifdef FE_UPWARD
      if (!fesetround (FE_UPWARD))
	{
	  result |= test_hex_in_one_mode (hex_tests[i].d, hex_tests[i].fmt,
					  hex_tests[i].ru, "FE_UPWARD");
	  fesetround (save_round_mode);
	}
#endif
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
