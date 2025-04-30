/* Test driver for nl_langinfo[_l] functions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>.

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

#include <langinfo.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>


struct map
{
  const char *str;
  int val;
} map[] =
{
#define VAL(name) { #name, name }
  VAL (ABDAY_1),
  VAL (ABDAY_2),
  VAL (ABDAY_3),
  VAL (ABDAY_4),
  VAL (ABDAY_5),
  VAL (ABDAY_6),
  VAL (ABDAY_7),
  VAL (ABMON_1),
  VAL (ABMON_10),
  VAL (ABMON_11),
  VAL (ABMON_12),
  VAL (ABMON_2),
  VAL (ABMON_3),
  VAL (ABMON_4),
  VAL (ABMON_5),
  VAL (ABMON_6),
  VAL (ABMON_7),
  VAL (ABMON_8),
  VAL (ABMON_9),
  VAL (ALT_DIGITS),
  VAL (ALTMON_1),
  VAL (ALTMON_10),
  VAL (ALTMON_11),
  VAL (ALTMON_12),
  VAL (ALTMON_2),
  VAL (ALTMON_3),
  VAL (ALTMON_4),
  VAL (ALTMON_5),
  VAL (ALTMON_6),
  VAL (ALTMON_7),
  VAL (ALTMON_8),
  VAL (ALTMON_9),
  VAL (AM_STR),
  VAL (CRNCYSTR),
  VAL (CURRENCY_SYMBOL),
  VAL (DAY_1),
  VAL (DAY_2),
  VAL (DAY_3),
  VAL (DAY_4),
  VAL (DAY_5),
  VAL (DAY_6),
  VAL (DAY_7),
  VAL (DECIMAL_POINT),
  VAL (D_FMT),
  VAL (D_T_FMT),
  VAL (ERA),
  VAL (ERA_D_FMT),
  VAL (ERA_D_T_FMT),
  VAL (ERA_T_FMT),
  VAL (ERA_YEAR),
  VAL (FRAC_DIGITS),
  VAL (GROUPING),
  VAL (INT_CURR_SYMBOL),
  VAL (INT_FRAC_DIGITS),
  VAL (MON_1),
  VAL (MON_10),
  VAL (MON_11),
  VAL (MON_12),
  VAL (MON_2),
  VAL (MON_3),
  VAL (MON_4),
  VAL (MON_5),
  VAL (MON_6),
  VAL (MON_7),
  VAL (MON_8),
  VAL (MON_9),
  VAL (MON_DECIMAL_POINT),
  VAL (MON_GROUPING),
  VAL (MON_THOUSANDS_SEP),
  VAL (NEGATIVE_SIGN),
  VAL (NOEXPR),
  VAL (NOSTR),
  VAL (N_CS_PRECEDES),
  VAL (N_SEP_BY_SPACE),
  VAL (N_SIGN_POSN),
  VAL (PM_STR),
  VAL (POSITIVE_SIGN),
  VAL (P_CS_PRECEDES),
  VAL (P_SEP_BY_SPACE),
  VAL (P_SIGN_POSN),
  VAL (RADIXCHAR),
  VAL (THOUSANDS_SEP),
  VAL (THOUSEP),
  VAL (T_FMT),
  VAL (T_FMT_AMPM),
  VAL (YESEXPR),
  VAL (YESSTR)
};


static int
map_paramstr (const char *str)
{
  int low = 0;
  int high = sizeof (map) / sizeof (map[0]);

  while (low < high)
    {
      int med = (low + high) / 2;
      int cmpres;

      cmpres = strcmp (str, map[med].str);
      if (cmpres == 0)
	return map[med].val;
      else if (cmpres > 0)
	low = med + 1;
      else
	high = med;
    }

  return -1;
}


#ifdef DEBUG
# define REASON(str) printf ("\"%s\" ignored: %s\n", buf, str)
#else
# define REASON(str)
#endif

static int
do_test (void)
{
  int result = 0;

  while (! feof (stdin))
    {
      char buf[1000];
      char *rp;
      char *locale;
      char *paramstr;
      char *expected;
      int param;

      if (fgets (buf, sizeof (buf), stdin) == NULL)
	break;

      /* Split the fields.   There are three is them:
	 1. locale
	 2. langinfo() parameter
	 3. expected result; this can be a string with white space etc.
      */
      rp = buf;
      while (*rp == ' ' || *rp == '\t')
	++rp;

      if  (*rp == '#')
	{
	  /* It's a comment line.  Ignore it.  */
	  REASON ("comment");
	  continue;
	}
      locale = rp;

      while (*rp != '\0' && *rp != ' ' && *rp != '\t' && *rp != '\n')
	++rp;
      if (*rp == '\0' || *rp == '\n')
	{
	  /* Incomplete line.  */
	  REASON ("incomplete line");
	  continue;
	}
      *rp++ = '\0';

      while (*rp == ' ' || *rp == '\t')
	++rp;
      paramstr = rp;

      while (*rp != '\0' && *rp != ' ' && *rp != '\t' && *rp != '\n')
	++rp;
      if (*rp == '\0' || *rp == '\n')
	{
	  /* Incomplete line.  */
	  REASON ("incomplete line");
	  continue;
	}
      *rp++ = '\0';

      while (*rp == ' ' || *rp == '\t')
	++rp;

      if (*rp == '"')
	{
	  char *wp;

	  expected = wp = ++rp;
	  while (*rp != '"' && *rp != '\n' && *rp != '\0')
	    {
	      if (*rp == '\\')
		{
		  ++rp;
		  if (*rp == '\0')
		    break;
		  if (*rp >= '0' && *rp <= '9')
		    {
		      int val = *rp - '0';
		      if (rp[1] >= '0' && rp[1] <= '9')
			{
			  ++rp;
			  val *= 10;
			  val += *rp - '0';
			  if (rp[1] >= '0' && rp[1] <= '9')
			    {
			      ++rp;
			      val *= 10;
			      val += *rp - '0';
			    }
			}
		      *rp = val;
		    }
		}
	      *wp++ = *rp++;
	    }

	  if (*rp != '"')
	    {
	      REASON ("missing '\"'");
	      continue;
	    }

	  *wp = '\0';
	}
      else
	{
	  expected = rp;
	  while (*rp != '\0' && *rp != '\n')
	    ++rp;
	  *rp = '\0';
	}

      param = map_paramstr (paramstr);
      if (param == -1)
	{
	  /* Invalid parameter.  */
	  REASON ("invalid parameter");
	  continue;
	}

      result = test_locale (locale, paramstr, param, expected);
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
