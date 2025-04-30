/* Tests of C and POSIX locale contents.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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
#include <limits.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>


static int
run_test (const char *locname)
{
  struct lconv *lc;
  const char *str;
  const wchar_t *wstr;
  int result = 0;
  locale_t loc;

  /* ISO C stuff.  */
  lc = localeconv ();
  if (lc == NULL)
    {
      printf ("localeconv failed for locale %s\n", locname);
      result = 1;
    }
  else
    {
#define STRTEST(name, exp) \
      do								      \
	if (strcmp (lc->name, exp) != 0)				      \
	  {								      \
	    printf (#name " in locale %s wrong (is \"%s\", should be \"%s\")\n",\
		    locname, lc->name, exp);				      \
	    result = 1;							      \
	  }								      \
      while (0)
      STRTEST (decimal_point, ".");
      STRTEST (thousands_sep, "");
      STRTEST (grouping, "");
      STRTEST (mon_decimal_point, "");
      STRTEST (mon_thousands_sep, "");
      STRTEST (mon_grouping, "");
      STRTEST (positive_sign, "");
      STRTEST (negative_sign, "");
      STRTEST (currency_symbol, "");
      STRTEST (int_curr_symbol, "");

#define CHARTEST(name, exp) \
      do								      \
	if (lc->name != exp)						      \
	  {								      \
	    printf (#name " in locale %s wrong (is %d, should be %d)\n",      \
		    locname, lc->name, CHAR_MAX);			      \
	    result = 1;							      \
	  }								      \
      while (0)
      CHARTEST (frac_digits, CHAR_MAX);
      CHARTEST (p_cs_precedes, CHAR_MAX);
      CHARTEST (n_cs_precedes, CHAR_MAX);
      CHARTEST (p_sep_by_space, CHAR_MAX);
      CHARTEST (n_sep_by_space, CHAR_MAX);
      CHARTEST (p_sign_posn, CHAR_MAX);
      CHARTEST (n_sign_posn, CHAR_MAX);
      CHARTEST (int_frac_digits, CHAR_MAX);
      CHARTEST (int_p_cs_precedes, CHAR_MAX);
      CHARTEST (int_n_cs_precedes, CHAR_MAX);
      CHARTEST (int_p_sep_by_space, CHAR_MAX);
      CHARTEST (int_n_sep_by_space, CHAR_MAX);
      CHARTEST (int_p_sign_posn, CHAR_MAX);
      CHARTEST (int_n_sign_posn, CHAR_MAX);
    }

#undef STRTEST
#define STRTEST(name, exp) \
  str = nl_langinfo (name);						      \
  if (strcmp (str, exp) != 0)						      \
    {									      \
      printf ("nl_langinfo(" #name ") in locale %s wrong "		      \
	      "(is \"%s\", should be \"%s\")\n", locname, str, exp);	      \
      result = 1;							      \
    }
#define WSTRTEST(name, exp) \
  wstr = (wchar_t *) nl_langinfo (name);				      \
  if (wcscmp (wstr, exp) != 0)						      \
    {									      \
      printf ("nl_langinfo(" #name ") in locale %s wrong "		      \
	      "(is \"%S\", should be \"%S\")\n", locname, wstr, exp);	      \
      result = 1;							      \
    }

  /* Unix stuff.  */
  STRTEST (ABDAY_1, "Sun");
  STRTEST (ABDAY_2, "Mon");
  STRTEST (ABDAY_3, "Tue");
  STRTEST (ABDAY_4, "Wed");
  STRTEST (ABDAY_5, "Thu");
  STRTEST (ABDAY_6, "Fri");
  STRTEST (ABDAY_7, "Sat");
  STRTEST (DAY_1, "Sunday");
  STRTEST (DAY_2, "Monday");
  STRTEST (DAY_3, "Tuesday");
  STRTEST (DAY_4, "Wednesday");
  STRTEST (DAY_5, "Thursday");
  STRTEST (DAY_6, "Friday");
  STRTEST (DAY_7, "Saturday");
  STRTEST (ABMON_1, "Jan");
  STRTEST (ABMON_2, "Feb");
  STRTEST (ABMON_3, "Mar");
  STRTEST (ABMON_4, "Apr");
  STRTEST (ABMON_5, "May");
  STRTEST (ABMON_6, "Jun");
  STRTEST (ABMON_7, "Jul");
  STRTEST (ABMON_8, "Aug");
  STRTEST (ABMON_9, "Sep");
  STRTEST (ABMON_10, "Oct");
  STRTEST (ABMON_11, "Nov");
  STRTEST (ABMON_12, "Dec");
  STRTEST (MON_1, "January");
  STRTEST (MON_2, "February");
  STRTEST (MON_3, "March");
  STRTEST (MON_4, "April");
  STRTEST (MON_5, "May");
  STRTEST (MON_6, "June");
  STRTEST (MON_7, "July");
  STRTEST (MON_8, "August");
  STRTEST (MON_9, "September");
  STRTEST (MON_10, "October");
  STRTEST (MON_11, "November");
  STRTEST (MON_12, "December");
  STRTEST (AM_STR, "AM");
  STRTEST (PM_STR, "PM");
  STRTEST (D_T_FMT, "%a %b %e %H:%M:%S %Y");
  STRTEST (D_FMT, "%m/%d/%y");
  STRTEST (T_FMT, "%H:%M:%S");
  STRTEST (T_FMT_AMPM, "%I:%M:%S %p");
  STRTEST (ERA, "");
  STRTEST (ERA_D_FMT, "");
  STRTEST (ERA_T_FMT, "");
  STRTEST (ERA_D_T_FMT, "");
  STRTEST (ALT_DIGITS, "");

  STRTEST (RADIXCHAR, ".");
  STRTEST (THOUSEP, "");

  STRTEST (YESEXPR, "^[yY]");
  STRTEST (NOEXPR, "^[nN]");

  /* Extensions.  */
  WSTRTEST (_NL_WABDAY_1, L"Sun");
  WSTRTEST (_NL_WABDAY_2, L"Mon");
  WSTRTEST (_NL_WABDAY_3, L"Tue");
  WSTRTEST (_NL_WABDAY_4, L"Wed");
  WSTRTEST (_NL_WABDAY_5, L"Thu");
  WSTRTEST (_NL_WABDAY_6, L"Fri");
  WSTRTEST (_NL_WABDAY_7, L"Sat");
  WSTRTEST (_NL_WDAY_1, L"Sunday");
  WSTRTEST (_NL_WDAY_2, L"Monday");
  WSTRTEST (_NL_WDAY_3, L"Tuesday");
  WSTRTEST (_NL_WDAY_4, L"Wednesday");
  WSTRTEST (_NL_WDAY_5, L"Thursday");
  WSTRTEST (_NL_WDAY_6, L"Friday");
  WSTRTEST (_NL_WDAY_7, L"Saturday");
  WSTRTEST (_NL_WABMON_1, L"Jan");
  WSTRTEST (_NL_WABMON_2, L"Feb");
  WSTRTEST (_NL_WABMON_3, L"Mar");
  WSTRTEST (_NL_WABMON_4, L"Apr");
  WSTRTEST (_NL_WABMON_5, L"May");
  WSTRTEST (_NL_WABMON_6, L"Jun");
  WSTRTEST (_NL_WABMON_7, L"Jul");
  WSTRTEST (_NL_WABMON_8, L"Aug");
  WSTRTEST (_NL_WABMON_9, L"Sep");
  WSTRTEST (_NL_WABMON_10, L"Oct");
  WSTRTEST (_NL_WABMON_11, L"Nov");
  WSTRTEST (_NL_WABMON_12, L"Dec");
  WSTRTEST (_NL_WMON_1, L"January");
  WSTRTEST (_NL_WMON_2, L"February");
  WSTRTEST (_NL_WMON_3, L"March");
  WSTRTEST (_NL_WMON_4, L"April");
  WSTRTEST (_NL_WMON_5, L"May");
  WSTRTEST (_NL_WMON_6, L"June");
  WSTRTEST (_NL_WMON_7, L"July");
  WSTRTEST (_NL_WMON_8, L"August");
  WSTRTEST (_NL_WMON_9, L"September");
  WSTRTEST (_NL_WMON_10, L"October");
  WSTRTEST (_NL_WMON_11, L"November");
  WSTRTEST (_NL_WMON_12, L"December");
  WSTRTEST (_NL_WAM_STR, L"AM");
  WSTRTEST (_NL_WPM_STR, L"PM");
  WSTRTEST (_NL_WD_T_FMT, L"%a %b %e %H:%M:%S %Y");
  WSTRTEST (_NL_WD_FMT, L"%m/%d/%y");
  WSTRTEST (_NL_WT_FMT, L"%H:%M:%S");
  WSTRTEST (_NL_WT_FMT_AMPM, L"%I:%M:%S %p");
  WSTRTEST (_NL_WERA_D_FMT, L"");
  WSTRTEST (_NL_WERA_T_FMT, L"");
  WSTRTEST (_NL_WERA_D_T_FMT, L"");
  WSTRTEST (_NL_WALT_DIGITS, L"");

  STRTEST (_DATE_FMT, "%a %b %e %H:%M:%S %Z %Y");
  WSTRTEST (_NL_W_DATE_FMT, L"%a %b %e %H:%M:%S %Z %Y");

  STRTEST (INT_CURR_SYMBOL, "");
  STRTEST (CURRENCY_SYMBOL, "");
  STRTEST (MON_DECIMAL_POINT, "");
  STRTEST (MON_THOUSANDS_SEP, "");
  STRTEST (MON_GROUPING, "");
  STRTEST (POSITIVE_SIGN, "");
  STRTEST (NEGATIVE_SIGN, "");
  STRTEST (GROUPING, "");

  STRTEST (YESSTR, "");
  STRTEST (NOSTR, "");

  /* Test the new locale mechanisms.  */
  loc = newlocale (LC_ALL_MASK, locname, NULL);
  if (loc == NULL)
    {
      printf ("cannot create locale object for locale %s\n", locname);
      result = 1;
    }
  else
    {
      int c;

#undef STRTEST
#define STRTEST(name, exp) \
      str = nl_langinfo_l (name, loc);				      \
      if (strcmp (str, exp) != 0)					      \
	{								      \
	  printf ("nl_langinfo_l(" #name ") in locale %s wrong "	      \
		  "(is \"%s\", should be \"%s\")\n", locname, str, exp);      \
	  result = 1;							      \
	}
#undef WSTRTEST
#define WSTRTEST(name, exp) \
      wstr = (wchar_t *) nl_langinfo_l (name, loc);			      \
      if (wcscmp (wstr, exp) != 0)					      \
	{								      \
	  printf ("nl_langinfo_l(" #name ") in locale %s wrong "	      \
		  "(is \"%S\", should be \"%S\")\n", locname, wstr, exp);     \
	  result = 1;							      \
	}

      /* Unix stuff.  */
      STRTEST (ABDAY_1, "Sun");
      STRTEST (ABDAY_2, "Mon");
      STRTEST (ABDAY_3, "Tue");
      STRTEST (ABDAY_4, "Wed");
      STRTEST (ABDAY_5, "Thu");
      STRTEST (ABDAY_6, "Fri");
      STRTEST (ABDAY_7, "Sat");
      STRTEST (DAY_1, "Sunday");
      STRTEST (DAY_2, "Monday");
      STRTEST (DAY_3, "Tuesday");
      STRTEST (DAY_4, "Wednesday");
      STRTEST (DAY_5, "Thursday");
      STRTEST (DAY_6, "Friday");
      STRTEST (DAY_7, "Saturday");
      STRTEST (ABMON_1, "Jan");
      STRTEST (ABMON_2, "Feb");
      STRTEST (ABMON_3, "Mar");
      STRTEST (ABMON_4, "Apr");
      STRTEST (ABMON_5, "May");
      STRTEST (ABMON_6, "Jun");
      STRTEST (ABMON_7, "Jul");
      STRTEST (ABMON_8, "Aug");
      STRTEST (ABMON_9, "Sep");
      STRTEST (ABMON_10, "Oct");
      STRTEST (ABMON_11, "Nov");
      STRTEST (ABMON_12, "Dec");
      STRTEST (MON_1, "January");
      STRTEST (MON_2, "February");
      STRTEST (MON_3, "March");
      STRTEST (MON_4, "April");
      STRTEST (MON_5, "May");
      STRTEST (MON_6, "June");
      STRTEST (MON_7, "July");
      STRTEST (MON_8, "August");
      STRTEST (MON_9, "September");
      STRTEST (MON_10, "October");
      STRTEST (MON_11, "November");
      STRTEST (MON_12, "December");
      STRTEST (AM_STR, "AM");
      STRTEST (PM_STR, "PM");
      STRTEST (D_T_FMT, "%a %b %e %H:%M:%S %Y");
      STRTEST (D_FMT, "%m/%d/%y");
      STRTEST (T_FMT, "%H:%M:%S");
      STRTEST (T_FMT_AMPM, "%I:%M:%S %p");
      STRTEST (ERA, "");
      STRTEST (ERA_D_FMT, "");
      STRTEST (ERA_T_FMT, "");
      STRTEST (ERA_D_T_FMT, "");
      STRTEST (ALT_DIGITS, "");

      STRTEST (RADIXCHAR, ".");
      STRTEST (THOUSEP, "");

      STRTEST (YESEXPR, "^[yY]");
      STRTEST (NOEXPR, "^[nN]");

      /* Extensions.  */
      WSTRTEST (_NL_WABDAY_1, L"Sun");
      WSTRTEST (_NL_WABDAY_2, L"Mon");
      WSTRTEST (_NL_WABDAY_3, L"Tue");
      WSTRTEST (_NL_WABDAY_4, L"Wed");
      WSTRTEST (_NL_WABDAY_5, L"Thu");
      WSTRTEST (_NL_WABDAY_6, L"Fri");
      WSTRTEST (_NL_WABDAY_7, L"Sat");
      WSTRTEST (_NL_WDAY_1, L"Sunday");
      WSTRTEST (_NL_WDAY_2, L"Monday");
      WSTRTEST (_NL_WDAY_3, L"Tuesday");
      WSTRTEST (_NL_WDAY_4, L"Wednesday");
      WSTRTEST (_NL_WDAY_5, L"Thursday");
      WSTRTEST (_NL_WDAY_6, L"Friday");
      WSTRTEST (_NL_WDAY_7, L"Saturday");
      WSTRTEST (_NL_WABMON_1, L"Jan");
      WSTRTEST (_NL_WABMON_2, L"Feb");
      WSTRTEST (_NL_WABMON_3, L"Mar");
      WSTRTEST (_NL_WABMON_4, L"Apr");
      WSTRTEST (_NL_WABMON_5, L"May");
      WSTRTEST (_NL_WABMON_6, L"Jun");
      WSTRTEST (_NL_WABMON_7, L"Jul");
      WSTRTEST (_NL_WABMON_8, L"Aug");
      WSTRTEST (_NL_WABMON_9, L"Sep");
      WSTRTEST (_NL_WABMON_10, L"Oct");
      WSTRTEST (_NL_WABMON_11, L"Nov");
      WSTRTEST (_NL_WABMON_12, L"Dec");
      WSTRTEST (_NL_WMON_1, L"January");
      WSTRTEST (_NL_WMON_2, L"February");
      WSTRTEST (_NL_WMON_3, L"March");
      WSTRTEST (_NL_WMON_4, L"April");
      WSTRTEST (_NL_WMON_5, L"May");
      WSTRTEST (_NL_WMON_6, L"June");
      WSTRTEST (_NL_WMON_7, L"July");
      WSTRTEST (_NL_WMON_8, L"August");
      WSTRTEST (_NL_WMON_9, L"September");
      WSTRTEST (_NL_WMON_10, L"October");
      WSTRTEST (_NL_WMON_11, L"November");
      WSTRTEST (_NL_WMON_12, L"December");
      WSTRTEST (_NL_WAM_STR, L"AM");
      WSTRTEST (_NL_WPM_STR, L"PM");
      WSTRTEST (_NL_WD_T_FMT, L"%a %b %e %H:%M:%S %Y");
      WSTRTEST (_NL_WD_FMT, L"%m/%d/%y");
      WSTRTEST (_NL_WT_FMT, L"%H:%M:%S");
      WSTRTEST (_NL_WT_FMT_AMPM, L"%I:%M:%S %p");
      WSTRTEST (_NL_WERA_D_FMT, L"");
      WSTRTEST (_NL_WERA_T_FMT, L"");
      WSTRTEST (_NL_WERA_D_T_FMT, L"");
      WSTRTEST (_NL_WALT_DIGITS, L"");

      STRTEST (_DATE_FMT, "%a %b %e %H:%M:%S %Z %Y");
      WSTRTEST (_NL_W_DATE_FMT, L"%a %b %e %H:%M:%S %Z %Y");

      STRTEST (INT_CURR_SYMBOL, "");
      STRTEST (CURRENCY_SYMBOL, "");
      STRTEST (MON_DECIMAL_POINT, "");
      STRTEST (MON_THOUSANDS_SEP, "");
      STRTEST (MON_GROUPING, "");
      STRTEST (POSITIVE_SIGN, "");
      STRTEST (NEGATIVE_SIGN, "");
      STRTEST (GROUPING, "");

      STRTEST (YESSTR, "");
      STRTEST (NOSTR, "");

      /* Character class tests.  */
      for (c = 0; c < 128; ++c)
	{
#define CLASSTEST(name) \
	  if (is##name (c) != is##name##_l (c, loc))			      \
	    {								      \
	      printf ("is%s('\\%o') != is%s_l('\\%o')\n",		      \
		      #name, c, #name, c);				      \
	      result = 1;						      \
	    }
	  CLASSTEST (alnum);
	  CLASSTEST (alpha);
	  CLASSTEST (blank);
	  CLASSTEST (cntrl);
	  CLASSTEST (digit);
	  CLASSTEST (lower);
	  CLASSTEST (graph);
	  CLASSTEST (print);
	  CLASSTEST (punct);
	  CLASSTEST (space);
	  CLASSTEST (upper);
	  CLASSTEST (xdigit);

	  /* Character mapping tests.  */
#define MAPTEST(name) \
	  if (to##name (c) != to##name##_l (c, loc))			      \
	    {								      \
	      printf ("to%s('\\%o') != to%s_l('\\%o'): '\\%o' vs '\\%o'\n", \
		      #name, c, #name, c,				      \
		      to##name (c), to##name##_l (c, loc));		      \
	      result = 1;						      \
	    }
	  MAPTEST (lower);
	  MAPTEST (upper);
	}

      /* Character class tests, this time for wide characters.  Note that
	 this only works because we know that the internal encoding is
	 UCS4.  */
      for (c = 0; c < 128; ++c)
	{
#undef CLASSTEST
#define CLASSTEST(name) \
	  if (isw##name (c) != isw##name##_l (c, loc))		      \
	    {								      \
	      printf ("isw%s('\\%o') != isw%s_l('\\%o')\n",		      \
		      #name, c, #name, c);				      \
	      result = 1;						      \
	    }
	  CLASSTEST (alnum);
	  CLASSTEST (alpha);
	  CLASSTEST (blank);
	  CLASSTEST (cntrl);
	  CLASSTEST (digit);
	  CLASSTEST (lower);
	  CLASSTEST (graph);
	  CLASSTEST (print);
	  CLASSTEST (punct);
	  CLASSTEST (space);
	  CLASSTEST (upper);
	  CLASSTEST (xdigit);

	  /* Character mapping tests.  Note that
	     this only works because we know that the internal encoding is
	     UCS4.  */
#undef MAPTEST
#define MAPTEST(name) \
	  if (tow##name (c) != tow##name##_l (c, loc))		      \
	    {								      \
	      printf ("tow%s('\\%o') != tow%s_l('\\%o'): '\\%o' vs '\\%o'\n",\
		      #name, c, #name, c,				      \
		      tow##name (c), tow##name##_l (c, loc));		      \
	      result = 1;						      \
	    }
	  MAPTEST (lower);
	  MAPTEST (upper);
	}

      freelocale (loc);
    }

  return result;
}


static int
do_test (void)
{
  int result;

  /* First use the name "C".  */
  if (setlocale (LC_ALL, "C") == NULL)
    {
      puts ("cannot set C locale");
      result = 1;
    }
  else
    result = run_test ("C");

  /* Then the name "POSIX".  */
  if (setlocale (LC_ALL, "POSIX") == NULL)
    {
      puts ("cannot set POSIX locale");
      result = 1;
    }
  else
    result |= run_test ("POSIX");

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
