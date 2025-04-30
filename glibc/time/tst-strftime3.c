/* Data-driven tests for strftime/strptime.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.  This file is
   part of the GNU C Library.

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
#include <time.h>
#include <locale.h>
#include <wchar.h>

#include <support/check.h>
#include <array_length.h>
#include <libc-diag.h>

/* These exist for the convenience of writing the test data, because
   zero-based vs one-based.  */
typedef enum
  {
    Sun, Mon, Tue, Wed, Thu, Fri, Sat
  } WeekDay;

typedef enum
  {
    Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
  } Month;

typedef struct
{
  /* A descriptive name of the test.  */
  const char *name;

  /* The specific date and time to be tested.  */
  int y, m, d;
  WeekDay w;
  int hh, mm, ss;

  /* The locale under which the conversion is done.  */
  const char *locale;

  /* Format passed to strftime.  */
  const char *format;

  /* Expected data, NUL terminated.  */
  const char *printed;

} Data;

/* Notes:

   Years are full 4-digit years, the code compensates.  Likewise,
   use month and weekday enums (above) which are zero-based.

   The encoded strings are multibyte strings in the C locale which
   reflect the same binary data as the expected strings.  When you run
   the test, the strings are printed as-is to stdout, so if your
   terminal is set for the correct encoding, they'll be printed
   "correctly".  Put the Unicode codes and UTF-8 samples in the
   comments.

   For convenience, mis-matched strings are printed in
   paste-compatible format, raw text format, and Unicode format.  Use
   "" between a hex escape sequence (like \xe8) and a following hex
   digit which should be considered as a printable character.

   To verify text, save the correct text in a file, and use "od -tx1
   -tc file" to see the raw hex values.  */

const Data data[] = {

  { "Baseline test",
    2019, Mar, 27, Wed, 14,  3, 22, "en_US.ISO-8859-1", "%Y-%m-%d %T",
    "2019-03-27 14:03:22" },


  { "Japanese era change, BCE/CE, before transition",
    0, Dec, 31, Sun, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U7D00><U5143><U524D>01<U5E74> 紀元前01年 */
    "\xe7\xb4\x80\xe5\x85\x83\xe5\x89\x8d""01\xe5\xb9\xb4" },
  { "Japanese era change, BCE/CE, after transition",
    1, Jan,  1, Mon, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U897F><U66A6>01<U5E74> 西暦01年 */
    "\xe8\xa5\xbf\xe6\x9a\xa6""01\xe5\xb9\xb4" },

  { "Japanese era change, BCE/CE, before transition",
    0, Dec, 31, Sun, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U7D00><U5143><U524D>01<U5E74> 紀元前01年 */
    "\xb5\xaa\xb8\xb5\xc1\xb0""01\xc7\xaf" },
  { "Japanese era change, BCE/CE, after transition",
    1, Jan,  1, Mon, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U897F><U66A6>01<U5E74> 西暦01年 */
    "\xc0\xbe\xce\xf1""01\xc7\xaf" },


  { "Japanese era change, 1873, before transition",
    1872, Dec, 31, Tue, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U897F><U66A6>1872<U5E74> 西暦1872年 */
    "\xe8\xa5\xbf\xe6\x9a\xa6""1872\xe5\xb9\xb4" },
  { "Japanese era change, 1873, after transition",
    1873, Jan,  1, Wed, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U660E><U6CBB>06<U5E74> 明治06年 */
    "\xe6\x98\x8e\xe6\xb2\xbb""06\xe5\xb9\xb4" },


  { "Japanese era change, 1873, before transition",
    1872, Dec, 31, Tue, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U897F><U66A6>1872<U5E74> 西暦1872年 */
    "\xc0\xbe\xce\xf1""1872\xc7\xaf" },
  { "Japanese era change, 1873, after transition",
    1873, Jan,  1, Wed, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U660E><U6CBB>06<U5E74> 明治06年 */
    "\xcc\xc0\xbc\xa3""06\xc7\xaf" },


  { "Japanese era change, 1912, before transition year",
    1911, Dec, 31, Sun, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U660E><U6CBB>44<U5E74> 明治44年 */
    "\xe6\x98\x8e\xe6\xb2\xbb""44\xe5\xb9\xb4" },
  { "Japanese era change, 1912, start of transition year",
    1912, Jan,  1, Mon, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U660E><U6CBB>45<U5E74> 明治45年 */
    "\xe6\x98\x8e\xe6\xb2\xbb""45\xe5\xb9\xb4" },

  { "Japanese era change, 1912, before transition",
    1912, Jul, 29, Mon, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U660E><U6CBB>45<U5E74> 明治45年 */
    "\xe6\x98\x8e\xe6\xb2\xbb""45\xe5\xb9\xb4" },
  { "Japanese era change, 1912, after transition",
    1912, Jul, 30, Tue, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5927><U6B63><U5143><U5E74> 大正元年 */
    "\xe5\xa4\xa7\xe6\xad\xa3\xe5\x85\x83\xe5\xb9\xb4" },

  { "Japanese era change, 1912, before end of transition year",
    1912, Dec, 31, Tue, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5927><U6B63><U5143><U5E74> 大正元年 */
    "\xe5\xa4\xa7\xe6\xad\xa3\xe5\x85\x83\xe5\xb9\xb4" },
  { "Japanese era change, 1912, after transition year",
    1913, Jan,  1, Wed, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5927><U6B63>02<U5E74> 大正02年 */
    "\xe5\xa4\xa7\xe6\xad\xa3""02\xe5\xb9\xb4" },


  { "Japanese era change, 1912, before transition year",
    1911, Dec, 31, Sun, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U660E><U6CBB>44<U5E74> 明治44年 */
    "\xcc\xc0\xbc\xa3""44\xc7\xaf" },
  { "Japanese era change, 1912, start of transition year",
    1912, Jan,  1, Mon, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U660E><U6CBB>45<U5E74> 明治45年 */
    "\xcc\xc0\xbc\xa3""45\xc7\xaf" },

  { "Japanese era change, 1912, before transition",
    1912, Jul, 29, Mon, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U660E><U6CBB>45<U5E74> 明治45年 */
    "\xcc\xc0\xbc\xa3""45\xc7\xaf" },
  { "Japanese era change, 1912, after transition",
    1912, Jul, 30, Tue, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5927><U6B63><U5143><U5E74> 大正元年 */
    "\xc2\xe7\xc0\xb5\xb8\xb5\xc7\xaf" },

  { "Japanese era change, 1912, before end of transition year",
    1912, Dec, 31, Tue, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5927><U6B63><U5143><U5E74> 大正元年 */
    "\xc2\xe7\xc0\xb5\xb8\xb5\xc7\xaf" },
  { "Japanese era change, 1912, after transition year",
    1913, Jan,  1, Wed, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5927><U6B63>02<U5E74> 大正02年 */
    "\xc2\xe7\xc0\xb5""02\xc7\xaf" },


  { "Japanese era change, 1926, before transition year",
    1925, Dec, 31, Thu, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5927><U6B63>14<U5E74> 大正14年 */
    "\xe5\xa4\xa7\xe6\xad\xa3""14\xe5\xb9\xb4" },
  { "Japanese era change, 1926, start of transition year",
    1926, Jan,  1, Fri, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5927><U6B63>15<U5E74> 大正15年 */
    "\xe5\xa4\xa7\xe6\xad\xa3""15\xe5\xb9\xb4" },

  { "Japanese era change, 1926, before transition",
    1926, Dec, 24, Fri, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5927><U6B63>15<U5E74> 大正15年 */
    "\xe5\xa4\xa7\xe6\xad\xa3""15\xe5\xb9\xb4" },
  { "Japanese era change, 1926, after transition",
    1926, Dec, 25, Sat, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U662D><U548C><U5143><U5E74> 昭和元年 */
    "\xe6\x98\xad\xe5\x92\x8c\xe5\x85\x83\xe5\xb9\xb4" },

  { "Japanese era change, 1926, before end of transition year",
    1926, Dec, 31, Fri, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U662D><U548C><U5143><U5E74> 昭和元年 */
    "\xe6\x98\xad\xe5\x92\x8c\xe5\x85\x83\xe5\xb9\xb4" },
  { "Japanese era change, 1926, after transition year",
    1927, Jan,  1, Sat, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /*  <U662D><U548C>02<U5E74> 昭和02年 */
    "\xe6\x98\xad\xe5\x92\x8c""02\xe5\xb9\xb4" },


  { "Japanese era change, 1926, before transition year",
    1925, Dec, 31, Thu, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5927><U6B63>14<U5E74> 大正14年 */
    "\xc2\xe7\xc0\xb5""14\xc7\xaf" },
  { "Japanese era change, 1926, start of transition year",
    1926, Jan,  1, Fri, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5927><U6B63>15<U5E74> 大正15年 */
    "\xc2\xe7\xc0\xb5""15\xc7\xaf" },

  { "Japanese era change, 1926, before transition",
    1926, Dec, 24, Fri, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5927><U6B63>15<U5E74> 大正15年 */
    "\xc2\xe7\xc0\xb5""15\xc7\xaf" },
  { "Japanese era change, 1926, after transition",
    1926, Dec, 25, Sat, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U662D><U548C><U5143><U5E74> 昭和元年 */
    "\xbe\xbc\xcf\xc2\xb8\xb5\xc7\xaf" },

  { "Japanese era change, 1926, before end of transition year",
    1926, Dec, 31, Fri, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U662D><U548C><U5143><U5E74> 昭和元年 */
    "\xbe\xbc\xcf\xc2\xb8\xb5\xc7\xaf" },
  { "Japanese era change, 1926, after transition year",
    1927, Jan,  1, Sat, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /*  <U662D><U548C>02<U5E74> 昭和02年 */
    "\xbe\xbc\xcf\xc2""02\xc7\xaf" },


  { "Japanese era change, 1989, before transition year",
    1988, Dec, 31, Sat, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U662D><U548C>63<U5E74> 昭和63年 */
    "\xe6\x98\xad\xe5\x92\x8c""63\xe5\xb9\xb4" },
  { "Japanese era change, 1989, start of transition year",
    1989, Jan,  1, Sun, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U662D><U548C>64<U5E74> 昭和64年 */
    "\xe6\x98\xad\xe5\x92\x8c""64\xe5\xb9\xb4" },

  { "Japanese era change, 1989, before transition",
    1989, Jan,  7, Sat, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U662D><U548C>64<U5E74> 昭和64年 */
    "\xe6\x98\xad\xe5\x92\x8c""64\xe5\xb9\xb4" },
  { "Japanese era change, 1989, after transition",
    1989, Jan,  8, Sun, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5E73><U6210><U5143><U5E74> 平成元年 */
    "\xe5\xb9\xb3\xe6\x88\x90\xe5\x85\x83\xe5\xb9\xb4" },

  { "Japanese era change, 1989, end of transition year",
    1989, Dec, 31, Sun, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5E73><U6210><U5143><U5E74> 平成元年 */
    "\xe5\xb9\xb3\xe6\x88\x90\xe5\x85\x83\xe5\xb9\xb4" },
  { "Japanese era change, 1989, after transition year",
    1990, Jan,  1, Mon, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5E73><U6210>02<U5E74> 平成02年 */
    "\xe5\xb9\xb3\xe6\x88\x90""02\xe5\xb9\xb4" },


  { "Japanese era change, 1989, before transition year",
    1988, Dec, 31, Sat, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U662D><U548C>63<U5E74> 昭和63年 */
    "\xbe\xbc\xcf\xc2""63\xc7\xaf" },
  { "Japanese era change, 1989, start of transition year",
    1989, Jan,  1, Sun, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U662D><U548C>64<U5E74> 昭和64年 */
    "\xbe\xbc\xcf\xc2""64\xc7\xaf" },

  { "Japanese era change, 1989, before transition",
    1989, Jan,  7, Sat, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U662D><U548C>64<U5E74> 昭和64年 */
    "\xbe\xbc\xcf\xc2""64\xc7\xaf" },
  { "Japanese era change, 1989, after transition",
    1989, Jan,  8, Sun, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5E73><U6210><U5143><U5E74> 平成元年 */
    "\xca\xbf\xc0\xae\xb8\xb5\xc7\xaf" },

  { "Japanese era change, 1989, end of transition year",
    1989, Dec, 31, Sun, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5E73><U6210><U5143><U5E74> 平成元年 */
    "\xca\xbf\xc0\xae\xb8\xb5\xc7\xaf" },
  { "Japanese era change, 1989, after transition year",
    1990, Jan,  1, Mon, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5E73><U6210>02<U5E74> 平成02年 */
    "\xca\xbf\xc0\xae""02\xc7\xaf" },


  { "Japanese era change, 2019, before transition year",
    2018, Dec, 31, Mon, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5E73><U6210>30<U5E74> 昭和30年 */
    "\xe5\xb9\xb3\xe6\x88\x90""30\xe5\xb9\xb4" },
  { "Japanese era change, 2019, start of transition year",
    2019, Jan,  1, Tue, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5E73><U6210>30<U5E74> 昭和31年 */
    "\xe5\xb9\xb3\xe6\x88\x90""31\xe5\xb9\xb4" },

  { "Japanese era change, 2019, before transition",
    2019, Apr, 30, Tue, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U5E73><U6210>30<U5E74> 昭和31年 */
    "\xe5\xb9\xb3\xe6\x88\x90""31\xe5\xb9\xb4" },
  { "Japanese era change, 2019, after transition",
    2019, May,  1, Wed, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U4EE4><U548C><U5143><U5E74> 令和元年 */
    "\xe4\xbb\xa4\xe5\x92\x8c\xe5\x85\x83\xe5\xb9\xb4" },

  { "Japanese era change, 2019, end of transition year",
    2019, Dec, 31, Tue, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U4EE4><U548C><U5143><U5E74> 令和元年 */
    "\xe4\xbb\xa4\xe5\x92\x8c\xe5\x85\x83\xe5\xb9\xb4" },
  { "Japanese era change, 2019, after transition year",
    2020, Jan,  1, Wed, 12, 00, 00, "ja_JP.UTF-8", "%EY",
    /* <U4EE4><U548C>02<U5E74> 令和02年 */
    "\xe4\xbb\xa4\xe5\x92\x8c""02\xe5\xb9\xb4" },


  { "Japanese era change, 2019, before transition year",
    2018, Dec, 31, Mon, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5E73><U6210>30<U5E74> 昭和30年 */
    "\xca\xbf\xc0\xae""30\xc7\xaf" },
  { "Japanese era change, 2019, start of transition year",
    2019, Jan,  1, Tue, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5E73><U6210>30<U5E74> 昭和31年 */
    "\xca\xbf\xc0\xae""31\xc7\xaf" },

  { "Japanese era change, 2019, before transition",
    2019, Apr, 30, Tue, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U5E73><U6210>30<U5E74> 昭和31年 */
    "\xca\xbf\xc0\xae""31\xc7\xaf" },
  { "Japanese era change, 2019, after transition",
    2019, May,  1, Wed, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U4EE4><U548C><U5143><U5E74> 令和元年 */
    "\xce\xe1\xcf\xc2\xb8\xb5\xc7\xaf" },

  { "Japanese era change, 2019, end of transition year",
    2019, Dec, 31, Tue, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U4EE4><U548C><U5143><U5E74> 令和元年 */
    "\xce\xe1\xcf\xc2\xb8\xb5\xc7\xaf" },
  { "Japanese era change, 2019, after transition year",
    2020, Jan,  1, Wed, 12, 00, 00, "ja_JP.EUC-JP", "%EY",
    /* <U4EE4><U548C>02<U5E74> 令和02年 */
    "\xce\xe1\xcf\xc2""02\xc7\xaf" },
};

#define NDATA array_length(data)

/* Size of buffer passed to strftime.  */
#define STRBUFLEN 1000
/* Size of buffer passed to tm_to_printed.  */
#define TMBUFLEN 50

/* Helper function to compare strings and print out mismatches in a
   format suitable for maintaining this test.  TEST_COMPARE_STRINGS
   prints out a less suitable format.  */

static void
print_string_hex (const char *header, const char *str)
{
  int tictoc = 0;
  const char *s = str;
  wchar_t w[STRBUFLEN];
  size_t i, wlen;

  printf ("%s : ", header);

  if (str == NULL)
    {
      printf ("<NULL>\n");
      return;
    }

  while (*s)
    {
      /* isgraph equivalent, but independent of current locale.  */
      if (' ' <= *s && *s <= '~')
	putchar (*s);
      else
	{
	  if (tictoc)
	    printf ("\033[36m");
	  else
	    printf ("\033[31m");
	  tictoc = ! tictoc;

	  printf ("\\x%02x\033[0m", (unsigned char) *s);
	}

      ++ s;
    }
  printf (" - %s\n", str);

  s = str;
  wlen = mbsrtowcs (w, &s, strlen (s), NULL);
  printf ("%*s", (int) strlen (header) + 3, " ");
  for (i = 0; i < wlen && i < strlen (str); i ++)
    {
      if (' ' <= w[i] && w[i] <= '~')
	putchar (w[i]);
      else
	printf ("<U%04X>", (int) w[i]);
    }
  printf ("\n");
}

static void
compare_strings (const char *got, const char *expected,
		 const char *filename, int lineno)
{
  if (got && expected && strcmp (got, expected) == 0)
    return;
  support_record_failure ();
  printf ("%s:%d: error: strftime output incorrect\n", filename, lineno);
  print_string_hex ("Got", got);
  print_string_hex ("Exp", expected);
}
#define COMPARE_STRINGS(g,e) compare_strings (g, e, __FILE__, __LINE__)

const char *weekday_name[] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri",
			       "Sat" };

/* Helper function to create a printable version of struct tm.  */
static void
tm_to_printed (struct tm *tm, char *buffer)
{
  const char *wn;
  char temp[50];

  if (0 <= tm->tm_wday && tm->tm_wday <= 6)
    wn = weekday_name[tm->tm_wday];
  else
    {
      wn = temp;
      sprintf (temp, "%d", tm->tm_wday);
    }

  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (9, 0)
  /* GCC 9 warns that strncmp may truncate its output, but that's why
     we're using it.  When it needs to truncate, it got corrupted
     data, and we only care that the string is different than valid
     data, which won't truncate.  */
  DIAG_IGNORE_NEEDS_COMMENT (9, "-Wformat-truncation=");
#endif
  snprintf (buffer, TMBUFLEN, "%04d/%02d/%02d %02d:%02d:%02d %s",
	    tm->tm_year + 1900,
	    tm->tm_mon + 1,
	    tm->tm_mday,
	    tm->tm_hour,
	    tm->tm_min,
	    tm->tm_sec,
	    wn);
  DIAG_POP_NEEDS_COMMENT;
}

static int
do_test (void)
{
  int i;
  char buffer[STRBUFLEN];
  char expected_time[TMBUFLEN];
  char got_time[TMBUFLEN];

  for (i = 0; i < NDATA; i ++)
    {
      const Data *d = &(data[i]);
      struct tm tm;
      struct tm tm2;
      size_t rv;
      char *rvp;

      /* Print this just to help debug failures.  */
      printf ("%s:\n\t%s %s %s\n", d->name, d->locale, d->format, d->printed);

      tm.tm_year = d->y - 1900;
      tm.tm_mon = d->m;
      tm.tm_mday = d->d;
      tm.tm_wday = d->w;
      tm.tm_hour = d->hh;
      tm.tm_min = d->mm;
      tm.tm_sec = d->ss;
      tm.tm_isdst = -1;

      /* LC_ALL may interfere with the snprintf in tm_to_printed.  */
      if (setlocale (LC_TIME, d->locale) == NULL)
	{
	  /* See the LOCALES list in the Makefile.  */
	  printf ("locale %s does not exist!\n", d->locale);
	  exit (EXIT_FAILURE);
	}
      /* This is just for printing wide characters if there's an error.  */
      setlocale (LC_CTYPE, d->locale);

      rv = strftime (buffer, sizeof (buffer), d->format, &tm);

      TEST_COMPARE (rv, strlen (d->printed));
      COMPARE_STRINGS (buffer, d->printed);

      /* Copy the original time, so that any fields not affected by
	 the call to strptime will match.  */
      tm2 = tm;

      rvp = strptime (d->printed, d->format, &tm2);

      TEST_COMPARE_STRING (rvp, "");

      tm_to_printed (&tm, expected_time);
      tm_to_printed (&tm2, got_time);
      TEST_COMPARE_STRING (got_time, expected_time);
    }

  return 0;
}

#include <support/test-driver.c>
