/* Test for regs allocation in re_search and re_match.
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
  struct re_registers regs;
  const char *s;
  int match, n;
  int result = 0;

  memset (&regex, '\0', sizeof (regex));
  regs.start = regs.end = NULL;
  regs.num_regs = 0;
  s = re_compile_pattern ("a", 1, &regex);
  if (s != NULL)
    {
      puts ("failed to compile pattern \"a\"");
      result = 1;
    }
  else
    {
      match = re_search (&regex, "baobab", 6, 0, 6, &regs);
      n = 1;
      if (match != 1)
	{
	  printf ("re_search returned %d, expected 1\n", match);
	  result = 1;
	}
      else if (regs.num_regs <= n || regs.start[n] != -1 || regs.end[n] != -1)
	{
	  puts ("re_search failed to fill the -1 sentinel");
	  result = 1;
	}
    }

  free (regex.buffer);
  memset (&regex, '\0', sizeof (regex));

  s = re_compile_pattern ("\\(\\(\\(a\\)\\)\\)", 13, &regex);
  if (s != NULL)
    {
      puts ("failed to compile pattern /\\(\\(\\(a\\)\\)\\)/");
      result = 1;
    }
  else
    {
      match = re_match (&regex, "apl", 3, 0, &regs);
      n = 4;
      if (match != 1)
	{
	  printf ("re_match returned %d, expected 1\n", match);
	  result = 1;
	}
      else if (regs.num_regs <= n || regs.start[n] != -1 || regs.end[n] != -1)
	{
	  puts ("re_match failed to fill the -1 sentinel");
	  result = 1;
	}
    }

  if (result == 0)
    puts (" -> OK");

  return result;
}
