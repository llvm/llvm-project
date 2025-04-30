/* Test for re_match with non-zero start.
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
#include <regex.h>

int
main (void)
{
  struct re_pattern_buffer regex;
  struct re_registers regs;
  const char *s;
  int match;
  int result = 0;

  regs.num_regs = 1;
  memset (&regex, '\0', sizeof (regex));
  s = re_compile_pattern ("[abc]*d", 7, &regex);
  if (s != NULL)
    {
      puts ("re_compile_pattern return non-NULL value");
      result = 1;
    }
  else
    {
      match = re_match (&regex, "foacabdxy", 9, 2, &regs);
      if (match != 5)
	{
	  printf ("re_match returned %d, expected 5\n", match);
	  result = 1;
	}
      else if (regs.start[0] != 2 || regs.end[0] != 7)
	{
	  printf ("re_match returned %d..%d, expected 2..7\n",
		  regs.start[0], regs.end[0]);
	  result = 1;
	}
	puts (" -> OK");
    }

  return result;
}
