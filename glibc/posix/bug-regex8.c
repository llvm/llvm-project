/* Test for the STOP parameter of re_match_2 and re_search_2.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Stepan Kasal <kasal@math.cas.cz>, 2002.

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
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <regex.h>


int
main (void)
{
  struct re_pattern_buffer regex;
  const char *s;
  int match[4];

  memset (&regex, '\0', sizeof (regex));

  s = re_compile_pattern ("xy$", 3, &regex);
  if (s != NULL)
    {
      puts ("failed to compile pattern \"xy$\"");
      return 1;
    }
  else
    match[0] = re_match_2(&regex,"xyz",3,NULL,0,0,NULL,2);

  free (regex.buffer);
  memset (&regex, '\0', sizeof (regex));

  s = re_compile_pattern ("xy\\>", 4, &regex);
  if (s != NULL)
    {
      puts ("failed to compile pattern \"xy\\>\"");
      return 1;
    }
  else
    match[1] = re_search_2(&regex,"xyz",3,NULL,0,0,2,NULL,2);

  free (regex.buffer);
  memset (&regex, '\0', sizeof (regex));

  s = re_compile_pattern ("xy \\<", 5, &regex);
  if (s != NULL)
    {
      puts ("failed to compile pattern \"xy \\<\"");
      return 1;
    }
  else
    {
      match[2] = re_match_2(&regex,"xy  ",4,NULL,0,0,NULL,3);
      match[3] = re_match_2(&regex,"xy z",4,NULL,0,0,NULL,3);
    }

  if (match[0] != -1 || match[1] != -1 || match[2] != -1 || match[3] != 3)
    {
      printf ("re_{match,search}_2 returned %d,%d,%d,%d, expected -1,-1,-1,3\n",
		match[0], match[1], match[2], match[3]);
      return 1;
    }

  puts (" -> OK");

  return 0;
}
