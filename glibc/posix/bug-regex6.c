/* Test for regexec.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2002.

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

#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <regex.h>


int
main (int argc, char *argv[])
{
  regex_t re;
  regmatch_t mat[10];
  int i, j, ret = 0;
  const char *locales[] = { "C", "de_DE.UTF-8" };
  const char *string = "http://www.regex.com/pattern/matching.html#intro";
  regmatch_t expect[10] = {
    { 0, 48 }, { 0, 5 }, { 0, 4 }, { 5, 20 }, { 7, 20 }, { 20, 42 },
    { -1, -1 }, { -1, -1 }, { 42, 48 }, { 43, 48 } };

  for (i = 0; i < sizeof (locales) / sizeof (locales[0]); ++i)
    {
      if (setlocale (LC_ALL, locales[i]) == NULL)
	{
	  puts ("cannot set locale");
	  ret = 1;
	}
      else if (regcomp (&re,
			"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?",
			REG_EXTENDED) != REG_NOERROR)
	{
	  puts ("cannot compile the regular expression");
	  ret = 1;
	}
      else if (regexec (&re, string, 10, mat, 0) == REG_NOMATCH)
	{
	  puts ("no match");
	  ret = 1;
	}
      else
	{
	  if (! memcmp (mat, expect, sizeof (mat)))
	    printf ("matching ok for %s locale\n", locales[i]);
	  else
	    {
	      printf ("matching failed for %s locale:\n", locales[i]);
	      ret = 1;
	      for (j = 0; j < 9; ++j)
		if (mat[j].rm_so != -1)
		  printf ("%d: %.*s\n", j, mat[j].rm_eo - mat[j].rm_so,
			  string + mat[j].rm_so);
	    }
	}
    }

  return ret;
}
