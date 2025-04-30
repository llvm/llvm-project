/* Test for re_search_2.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2001.

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
#include <regex.h>

int
main (void)
{
  struct re_pattern_buffer regex;
  const char *s;
  int match[3];
  int result = 0;

  memset (&regex, '\0', sizeof (regex));

  setlocale (LC_ALL, "C");

  s = re_compile_pattern ("ab[cde]", 7, &regex);
  if (s != NULL)
    {
      puts ("re_compile_pattern returned non-NULL value");
      result = 1;
    }
  else
    {
      match[0] = re_search_2 (&regex, "xyabez", 6, "", 0, 1, 5, NULL, 6);
      match[1] = re_search_2 (&regex, NULL, 0, "abc", 3, 0, 3, NULL, 3);
      match[2] = re_search_2 (&regex, "xya", 3, "bd", 2, 2, 3, NULL, 5);
      if (match[0] != 2 || match[1] != 0 || match[2] != 2)
	{
	  printf ("re_search_2 returned %d,%d,%d, expected 2,0,2\n",
		  match[0], match[1], match[2]);
	  result = 1;
	}
      else
	puts (" -> OK");
    }

  return result;
}
