/* Test for memory handling in regex.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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


static const char text[] = "#! /bin/sh";

int
main (void)
{
  regex_t re;
  regmatch_t rm[2];
  int n;

  mtrace ();

  n = regcomp (&re, "^#! */.*/(k|ba||pdk|z)sh", REG_EXTENDED);
  if (n != 0)
    {
      char buf[500];
      regerror (n, &re, buf, sizeof (buf));
      printf ("regcomp failed: %s\n", buf);
      exit (1);
    }

  for (n = 0; n < 20; ++n)
    {
      if (regexec (&re, text, 2, rm, 0))
	{
	  puts ("regexec failed");
	  exit (2);
	}
      if (rm[0].rm_so != 0 || rm[0].rm_eo != 10
	  || rm[1].rm_so != 8 || rm[1].rm_eo != 8)
	{
	  printf ("regexec match failure: %d %d %d %d\n",
		  rm[0].rm_so, rm[0].rm_eo, rm[1].rm_so, rm[1].rm_eo);
	  exit (3);
	}
    }

  regfree (&re);

  return 0;
}
