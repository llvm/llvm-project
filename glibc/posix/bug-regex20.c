/* Test for UTF-8 regular expression optimizations.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <sys/types.h>
#include <mcheck.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>

#define RE_NO_INTERNAL_PROTOTYPES 1
#include "regex_internal.h"

#define BRE RE_SYNTAX_POSIX_BASIC
#define ERE RE_SYNTAX_POSIX_EXTENDED

static struct
{
  int syntax;
  const char *pattern;
  const char *string;
  int res, optimize;
} tests[] = {
  /* \xc3\x84		LATIN CAPITAL LETTER A WITH DIAERESIS
     \xc3\x96		LATIN CAPITAL LETTER O WITH DIAERESIS
     \xc3\xa4		LATIN SMALL LETTER A WITH DIAERESIS
     \xc3\xb6		LATIN SMALL LETTER O WITH DIAERESIS
     \xe2\x80\x94	EM DASH  */
  /* Should be optimized.  */
  {BRE, "foo", "b\xc3\xa4rfoob\xc3\xa4z", 4, 1},
  {BRE, "b\xc3\xa4z", "b\xc3\xa4rfoob\xc3\xa4z", 7, 1},
  {BRE, "b\xc3\xa4*z", "b\xc3\xa4rfoob\xc3\xa4z", 7, 1},
  {BRE, "b\xc3\xa4*z", "b\xc3\xa4rfoobz", 7, 1},
  {BRE, "b\xc3\xa4\\+z", "b\xc3\xa4rfoob\xc3\xa4\xc3\xa4z", 7, 1},
  {BRE, "b\xc3\xa4\\?z", "b\xc3\xa4rfoob\xc3\xa4z", 7, 1},
  {BRE, "b\xc3\xa4\\{1,2\\}z", "b\xc3\xa4rfoob\xc3\xa4z", 7, 1},
  {BRE, "^x\\|xy*z$", "\xc3\xb6xyyz", 2, 1},
  {BRE, "^x\\\\y\\{6\\}z\\+", "x\\yyyyyyzz\xc3\xb6", 0, 1},
  {BRE, "^x\\\\y\\{2,36\\}z\\+", "x\\yzz\xc3\xb6", -1, 1},
  {BRE, "^x\\\\y\\{,3\\}z\\+", "x\\yyyzz\xc3\xb6", 0, 1},
  {BRE, "^x\\|x\xc3\xa4*z$", "\xc3\xb6x\xc3\xa4\xc3\xa4z", 2, 1},
  {BRE, "^x\\\\\xc3\x84\\{6\\}z\\+",
   "x\\\xc3\x84\xc3\x84\xc3\x84\xc3\x84\xc3\x84\xc3\x84zz\xc3\xb6", 0, 1},
  {BRE, "^x\\\\\xc3\x84\\{2,36\\}z\\+", "x\\\xc3\x84zz\xc3\xb6", -1, 1},
  {BRE, "^x\\\\\xc3\x84\\{,3\\}z\\+",
   "x\\\xc3\x84\xc3\x84\xc3\x84zz\xc3\xb6", 0, 1},
  {BRE, "x[C]y", "axCy", 1, 1},
  {BRE, "x[ABC]y", "axCy", 1, 1},
  {BRE, "\\`x\\|z\\'", "x\xe2\x80\x94", 0, 1},
  {BRE, "\\(xy\\)z\\1a\\1", "\xe2\x80\x94xyzxyaxy\xc3\x84", 3, 1},
  {BRE, "xy\\?z", "\xc3\x84xz\xc3\xb6", 2, 1},
  {BRE, "\\`\xc3\x84\\|z\\'", "\xc3\x84\xe2\x80\x94", 0, 1},
  {BRE, "\\(x\xc3\x84\\)z\\1\x61\\1",
   "\xe2\x80\x94x\xc3\x84zx\xc3\x84\x61x\xc3\x84\xc3\x96", 3, 1},
  {BRE, "x\xc3\x96\\?z", "\xc3\x84xz\xc3\xb6", 2, 1},
  {BRE, "x.y", "ax\xe2\x80\x94yz", 1, 1},
  {BRE, "x.*z", "\xc3\x84xz", 2, 1},
  {BRE, "x.*z", "\xc3\x84x\xe2\x80\x94z", 2, 1},
  {BRE, "x.*z", "\xc3\x84x\xe2\x80\x94y\xf1\x90\x80\x90z", 2, 1},
  {BRE, "x.*z", "\xc3\x84x\xe2\x80\x94\xc3\x94\xf1\x90\x80\x90z", 2, 1},
  {BRE, "x.\\?z", "axz", 1, 1},
  {BRE, "x.\\?z", "axyz", 1, 1},
  {BRE, "x.\\?z", "ax\xc3\x84z", 1, 1},
  {BRE, "x.\\?z", "ax\xe2\x80\x94z", 1, 1},
  {BRE, "x.\\?z", "ax\xf0\x9d\x80\x80z", 1, 1},
  {BRE, "x.\\?z", "ax\xf9\x81\x82\x83\x84z", 1, 1},
  {BRE, "x.\\?z", "ax\xfd\xbf\xbf\xbf\xbf\xbfz", 1, 1},
  {BRE, ".", "y", 0, 1},
  {BRE, ".", "\xc3\x84", 0, 1},
  {BRE, ".", "\xe2\x80\x94", 0, 1},
  {BRE, ".", "\xf0\x9d\x80\x80", 0, 1},
  {BRE, ".", "\xf9\x81\x82\x83\x84", 0, 1},
  {BRE, ".", "\xfd\xbf\xbf\xbf\xbf\xbf", 0, 1},
  {BRE, "x.\\?z", "axyyz", -1, 1},
  {BRE, "x.\\?z", "ax\xc3\x84\xc3\x96z", -1, 1},
  {BRE, "x.\\?z", "ax\xe2\x80\x94\xc3\xa4z", -1, 1},
  {BRE, "x.\\?z", "ax\xf0\x9d\x80\x80yz", -1, 1},
  {BRE, "x.\\?z", "ax\xf9\x81\x82\x83\x84\xf0\x9d\x80\x81z", -1, 1},
  {BRE, "x.\\?z", "ax\xfd\xbf\xbf\xbf\xbf\xbf\xc3\x96z", -1, 1},
  {BRE, "x.\\+z", "\xe2\x80\x94xz", -1, 1},
  {BRE, "x.\\+z", "\xe2\x80\x94xyz", 3, 1},
  {BRE, "x.\\+z", "\xe2\x80\x94x\xc3\x84y\xe2\x80\x94z", 3, 1},
  {BRE, "x.\\+z", "\xe2\x80\x94x\xe2\x80\x94z", 3, 1},
  {BRE, "x.\\+z", "\xe2\x80\x94x\xf0\x9d\x80\x80\xc3\x84z", 3, 1},
  {BRE, "x.\\+z", "\xe2\x80\x94x.~\xe2\x80\x94\xf9\x81\x82\x83\x84z", 3, 1},
  {BRE, "x.\\+z", "\xe2\x80\x94x\xfd\xbf\xbf\xbf\xbf\xbfz", 3, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94xz", -1, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94x\xc3\x96y\xc3\xa4z", -1, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94xyz", 3, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94x\xc3\x84\xe2\x80\x94z", 3, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94x\xe2\x80\x94z", 3, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94x\xf0\x9d\x80\x80\xc3\x84z", 3, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94x~\xe2\x80\x94z", 3, 1},
  {BRE, "x.\\{1,2\\}z", "\xe2\x80\x94x\xfd\xbf\xbf\xbf\xbf\xbfz", 3, 1},
  {BRE, "x\\(.w\\|\xc3\x86\\)\\?z", "axz", 1, 1},
  {BRE, "x\\(.w\\|\xc3\x86\\)\\?z", "ax\xfd\xbf\xbf\xbf\xbf\xbfwz", 1, 1},
  {BRE, "x\\(.w\\|\xc3\x86\\)\\?z", "ax\xc3\x86z", 1, 1},
  {BRE, "x\\(.w\\|\xc3\x86\\)\\?z", "ax\xe2\x80\x96wz", 1, 1},
  {ERE, "foo", "b\xc3\xa4rfoob\xc3\xa4z", 4, 1},
  {ERE, "^x|xy*z$", "\xc3\xb6xyyz", 2, 1},
  {ERE, "^x\\\\y{6}z+", "x\\yyyyyyzz\xc3\xb6", 0, 1},
  {ERE, "^x\\\\y{2,36}z+", "x\\yzz\xc3\xb6", -1, 1},
  {ERE, "^x\\\\y{,3}z+", "x\\yyyzz\xc3\xb6", 0, 1},
  {ERE, "x[C]y", "axCy", 1, 1},
  {ERE, "x[ABC]y", "axCy", 1, 1},
  {ERE, "\\`x|z\\'", "x\xe2\x80\x94", 0, 1},
  {ERE, "(xy)z\\1a\\1", "\xe2\x80\x94xyzxyaxy\xc3\x84", 3, 1},
  {ERE, "xy?z", "\xc3\x84xz\xc3\xb6", 2, 1},
  {ERE, "x.y", "ax\xe2\x80\x94yz", 1, 1},
  {ERE, "x.*z", "\xc3\x84xz", 2, 1},
  {ERE, "x.*z", "\xc3\x84x\xe2\x80\x94z", 2, 1},
  {ERE, "x.*z", "\xc3\x84x\xe2\x80\x94y\xf1\x90\x80\x90z", 2, 1},
  {ERE, "x.*z", "\xc3\x84x\xe2\x80\x94\xc3\x94\xf1\x90\x80\x90z", 2, 1},
  {ERE, "x.?z", "axz", 1, 1},
  {ERE, "x.?z", "axyz", 1, 1},
  {ERE, "x.?z", "ax\xc3\x84z", 1, 1},
  {ERE, "x.?z", "ax\xe2\x80\x94z", 1, 1},
  {ERE, "x.?z", "ax\xf0\x9d\x80\x80z", 1, 1},
  {ERE, "x.?z", "ax\xf9\x81\x82\x83\x84z", 1, 1},
  {ERE, "x.?z", "ax\xfd\xbf\xbf\xbf\xbf\xbfz", 1, 1},
  {ERE, "x.?z", "axyyz", -1, 1},
  {ERE, "x.?z", "ax\xc3\x84\xc3\x96z", -1, 1},
  {ERE, "x.?z", "ax\xe2\x80\x94\xc3\xa4z", -1, 1},
  {ERE, "x.?z", "ax\xf0\x9d\x80\x80yz", -1, 1},
  {ERE, "x.?z", "ax\xf9\x81\x82\x83\x84\xf0\x9d\x80\x81z", -1, 1},
  {ERE, "x.?z", "ax\xfd\xbf\xbf\xbf\xbf\xbf\xc3\x96z", -1, 1},
  {ERE, "x.+z", "\xe2\x80\x94xz", -1, 1},
  {ERE, "x.+z", "\xe2\x80\x94xyz", 3, 1},
  {ERE, "x.+z", "\xe2\x80\x94x\xc3\x84y\xe2\x80\x94z", 3, 1},
  {ERE, "x.+z", "\xe2\x80\x94x\xe2\x80\x94z", 3, 1},
  {ERE, "x.+z", "\xe2\x80\x94x\xf0\x9d\x80\x80\xc3\x84z", 3, 1},
  {ERE, "x.+z", "\xe2\x80\x94x.~\xe2\x80\x94\xf9\x81\x82\x83\x84z", 3, 1},
  {ERE, "x.+z", "\xe2\x80\x94x\xfd\xbf\xbf\xbf\xbf\xbfz", 3, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94xz", -1, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94x\xc3\x96y\xc3\xa4z", -1, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94xyz", 3, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94x\xc3\x84\xe2\x80\x94z", 3, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94x\xe2\x80\x94z", 3, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94x\xf0\x9d\x80\x80\xc3\x84z", 3, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94x~\xe2\x80\x94z", 3, 1},
  {ERE, "x.{1,2}z", "\xe2\x80\x94x\xfd\xbf\xbf\xbf\xbf\xbfz", 3, 1},
  {ERE, "x(.w|\xc3\x86)?z", "axz", 1, 1},
  {ERE, "x(.w|\xc3\x86)?z", "ax\xfd\xbf\xbf\xbf\xbf\xbfwz", 1, 1},
  {ERE, "x(.w|\xc3\x86)?z", "ax\xc3\x86z", 1, 1},
  {ERE, "x(.w|\xc3\x86)?z", "ax\xe2\x80\x96wz", 1, 1},
  /* Should not be optimized.  */
  {BRE, "x[\xc3\x84\xc3\xa4]y", "ax\xc3\xa4y", 1, 0},
  {BRE, "x[A-Z,]y", "axCy", 1, 0},
  {BRE, "x[^y]z", "ax\xe2\x80\x94z", 1, 0},
  {BRE, "x[[:alnum:]]z", "ax\xc3\x96z", 1, 0},
  {BRE, "x[[=A=]]z", "axAz", 1, 0},
  {BRE, "x[[=\xc3\x84=]]z", "ax\xc3\x84z", 1, 0},
  {BRE, "\\<g", "\xe2\x80\x94g", 3, 0},
  {BRE, "\\bg\\b", "\xe2\x80\x94g", 3, 0},
  {BRE, "\\Bg\\B", "\xc3\xa4g\xc3\xa4", 2, 0},
  {BRE, "a\\wz", "a\xc3\x84z", 0, 0},
  {BRE, "x\\Wz", "\xc3\x96x\xe2\x80\x94z", 2, 0},
  {ERE, "x[\xc3\x84\xc3\xa4]y", "ax\xc3\xa4y", 1, 0},
  {ERE, "x[A-Z,]y", "axCy", 1, 0},
  {ERE, "x[^y]z", "ax\xe2\x80\x94z", 1, 0},
  {ERE, "x[[:alnum:]]z", "ax\xc3\x96z", 1, 0},
  {ERE, "x[[=A=]]z", "axAz", 1, 0},
  {ERE, "x[[=\xc3\x84=]]z", "ax\xc3\x84z", 1, 0},
  {ERE, "\\<g", "\xe2\x80\x94g", 3, 0},
  {ERE, "\\bg\\b", "\xe2\x80\x94g", 3, 0},
  {ERE, "\\Bg\\B", "\xc3\xa4g\xc3\xa4", 2, 0},
  {ERE, "a\\wz", "a\xc3\x84z", 0, 0},
  {ERE, "x\\Wz", "\xc3\x96x\xe2\x80\x94z", 2, 0},
};

int
main (void)
{
  struct re_pattern_buffer regbuf;
  const char *err;
  size_t i;
  int ret = 0;

  mtrace ();

  setlocale (LC_ALL, "de_DE.UTF-8");
  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      int res, optimized;

      re_set_syntax (tests[i].syntax);
      memset (&regbuf, '\0', sizeof (regbuf));
      err = re_compile_pattern (tests[i].pattern, strlen (tests[i].pattern),
                                &regbuf);
      if (err != NULL)
	{
	  printf ("re_compile_pattern failed: %s\n", err);
	  ret = 1;
	  continue;
	}

      /* Check if re_search will be done as multi-byte or single-byte.  */
      optimized = ((re_dfa_t *) regbuf.buffer)->mb_cur_max == 1;
      if (optimized != tests[i].optimize)
        {
          printf ("pattern %zd %soptimized while it should%s be\n",
		  i, optimized ? "" : "not ", tests[i].optimize ? "" : " not");
	  ret = 1;
        }

      int str_len = strlen (tests[i].string);
      res = re_search (&regbuf, tests[i].string, str_len, 0, str_len, NULL);
      if (res != tests[i].res)
	{
	  printf ("re_search %zd failed: %d\n", i, res);
	  ret = 1;
	  regfree (&regbuf);
	  continue;
	}

      res = re_search (&regbuf, tests[i].string, str_len, str_len, -str_len,
		       NULL);
      if (res != tests[i].res)
	{
	  printf ("backward re_search %zd failed: %d\n", i, res);
	  ret = 1;
	  regfree (&regbuf);
	  continue;
	}
      regfree (&regbuf);

      re_set_syntax (tests[i].syntax | RE_ICASE);
      memset (&regbuf, '\0', sizeof (regbuf));
      err = re_compile_pattern (tests[i].pattern, strlen (tests[i].pattern),
                                &regbuf);
      if (err != NULL)
	{
	  printf ("re_compile_pattern failed: %s\n", err);
	  ret = 1;
	  continue;
	}

      /* Check if re_search will be done as multi-byte or single-byte.  */
      optimized = ((re_dfa_t *) regbuf.buffer)->mb_cur_max == 1;
      if (optimized)
        {
          printf ("pattern %zd optimized while it should not be when case insensitive\n",
		  i);
	  ret = 1;
        }

      res = re_search (&regbuf, tests[i].string, str_len, 0, str_len, NULL);
      if (res != tests[i].res)
	{
	  printf ("ICASE re_search %zd failed: %d\n", i, res);
	  ret = 1;
	  regfree (&regbuf);
	  continue;
	}

      res = re_search (&regbuf, tests[i].string, str_len, str_len, -str_len,
		       NULL);
      if (res != tests[i].res)
	{
	  printf ("ICASE backward re_search %zd failed: %d\n", i, res);
	  ret = 1;
	  regfree (&regbuf);
	  continue;
	}
      regfree (&regbuf);
    }

  return ret;
}
