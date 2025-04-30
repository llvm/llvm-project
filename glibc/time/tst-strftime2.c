/* Verify the behavior of strftime on alternative representation for
   year.

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

#include <array_length.h>
#include <stdbool.h>
#include <support/check.h>
#include <stdlib.h>
#include <locale.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

static const char *locales[] =
{
  "ja_JP.UTF-8", "lo_LA.UTF-8", "th_TH.UTF-8",
  "zh_TW.UTF-8", "cmn_TW.UTF-8", "hak_TW.UTF-8",
  "nan_TW.UTF-8", "lzh_TW.UTF-8"
};

/* Must match locale index into locales array.  */
enum
{
  ja_JP, lo_LA, th_TH,
  zh_TW, cmn_TW, hak_TW, nan_TW, lzh_TW
};

static const char *formats[] = { "%EY", "%_EY", "%-EY" };

typedef struct
{
  const int d, m, y;
} date_t;

static const date_t dates[] =
{
  {  1,  4, 1910 },
  { 31, 12, 1911 },
  {  1,  1, 1912 },
  {  1,  4, 1913 },
  {  1,  4, 1988 },
  {  7,  1, 1989 },
  {  8,  1, 1989 },
  {  1,  4, 1990 },
  {  1,  4, 1997 },
  {  1,  4, 1998 },
  {  1,  4, 2010 },
  {  1,  4, 2011 },
  { 30,  4, 2019 },
  {  1,  5, 2019 }
};

static char ref[array_length (locales)][array_length (formats)]
	       [array_length (dates)][100];

static bool
is_before (const int i, const int d, const int m, const int y)
{
  if (dates[i].y < y)
    return true;
  else if (dates[i].y > y)
    return false;
  else if (dates[i].m < m)
    return true;
  else if (dates[i].m > m)
    return false;
  else
    return dates[i].d < d;
}

static void
mkreftable (void)
{
  int i, j, k, yr;
  const char *era, *sfx;
  /* Japanese era year to be checked.  */
  static const int yrj[] =
  {
    43, 44, 45, 2,
    63, 64, 1, 2, 9, 10, 22, 23, 31, 1
  };
  /* Buddhist calendar year to be checked.  */
  static const int yrb[] =
  {
    2453, 2454, 2455, 2456,
    2531, 2532, 2532, 2533, 2540, 2541, 2553, 2554, 2562, 2562
  };
  /* R.O.C. calendar year to be checked.  Negative number is prior to
     Minguo counting up.  */
  static const int yrc[] =
  {
    -2, -1, 1, 2,
    77, 78, 78, 79, 86, 87, 99, 100, 108, 108
  };

  for (i = 0; i < array_length (locales); i++)
    for (j = 0; j < array_length (formats); j++)
      for (k = 0; k < array_length (dates); k++)
	{
	  if (i == ja_JP)
	    {
	      era = (is_before (k, 30,  7, 1912)) ? "\u660e\u6cbb"
		  : (is_before (k, 25, 12, 1926)) ? "\u5927\u6b63"
		  : (is_before (k,  8,  1, 1989)) ? "\u662d\u548c"
		  : (is_before (k,  1,  5, 2019)) ? "\u5e73\u6210"
						  : "\u4ee4\u548c";
	      yr = yrj[k], sfx = "\u5e74";
	    }
	  else if (i == lo_LA)
	    era = "\u0e9e.\u0eaa. ", yr = yrb[k], sfx = "";
	  else if (i == th_TH)
	    era = "\u0e1e.\u0e28. ", yr = yrb[k], sfx = "";
	  else if (i == zh_TW || i == cmn_TW || i == hak_TW
		   || i == nan_TW || i == lzh_TW)
	    {
	      era = (is_before (k, 1, 1, 1912)) ? "\u6c11\u524d"
						: "\u6c11\u570b";
	      yr = yrc[k], sfx = "\u5e74";
	    }
	  else
	    FAIL_EXIT1 ("Invalid table index!");
	  if (yr == 1)
	    sprintf (ref[i][j][k], "%s\u5143%s", era, sfx);
	  else if (j == 0)
	    sprintf (ref[i][j][k], "%s%02d%s", era, abs (yr), sfx);
	  else if (j == 1)
	    sprintf (ref[i][j][k], "%s%2d%s", era, abs (yr), sfx);
	  else if (j == 2)
	    sprintf (ref[i][j][k], "%s%d%s", era, abs (yr), sfx);
	  else
	    FAIL_EXIT1 ("Invalid table index!");
	}
}

static int
do_test (void)
{
  int i, j, k, result = 0;
  struct tm ttm;
  char date[11], buf[100];
  size_t r, e;

  mkreftable ();
  for (i = 0; i < array_length (locales); i++)
    {
      if (setlocale (LC_ALL, locales[i]) == NULL)
	{
	  printf ("locale %s does not exist, skipping...\n", locales[i]);
	  continue;
	}
      printf ("[%s]\n", locales[i]);
      for (j = 0; j < array_length (formats); j++)
	{
	  for (k = 0; k < array_length (dates); k++)
	    {
	      ttm.tm_mday = dates[k].d;
	      ttm.tm_mon  = dates[k].m - 1;
	      ttm.tm_year = dates[k].y - 1900;
	      strftime (date, sizeof (date), "%F", &ttm);
	      r = strftime (buf, sizeof (buf), formats[j], &ttm);
	      e = strlen (ref[i][j][k]);
	      printf ("%s\t\"%s\"\t\"%s\"", date, formats[j], buf);
	      if (strcmp (buf, ref[i][j][k]) != 0)
		{
		  printf ("\tshould be \"%s\"", ref[i][j][k]);
		  if (r != e)
		    printf ("\tgot: %zu, expected: %zu", r, e);
		  result = 1;
		}
	      else
		printf ("\tOK");
	      putchar ('\n');
	    }
	  putchar ('\n');
	}
    }
  return result;
}

#include <support/test-driver.c>
