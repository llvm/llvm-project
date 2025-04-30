/* Test re.translate != NULL.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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
#include <locale.h>
#include <regex.h>
#include <stdio.h>
#include <string.h>

int
main (void)
{
  struct re_pattern_buffer re;
  char trans[256];
  int i, result = 0;
  const char *s;

  setlocale (LC_ALL, "de_DE.ISO-8859-1");

  for (i = 0; i < 256; ++i)
    trans[i] = tolower (i);

  re_set_syntax (RE_SYNTAX_POSIX_EGREP);

  memset (&re, 0, sizeof (re));
  re.translate = (unsigned char *) trans;
  s = re_compile_pattern ("\\W", 2, &re);

  if (s != NULL)
    {
      printf ("failed to compile pattern \"\\W\": %s\n", s);
      result = 1;
    }
  else
    {
      int ret = re_search (&re, "abc.de", 6, 0, 6, NULL);
      if (ret != 3)
	{
	  printf ("1st re_search returned %d\n", ret);
	  result = 1;
	}

      ret = re_search (&re, "\xc4\xd6\xae\xf7", 4, 0, 4, NULL);
      if (ret != 2)
	{
	  printf ("2nd re_search returned %d\n", ret);
	  result = 1;
	}
      re.translate = NULL;
      regfree (&re);
    }

  memset (&re, 0, sizeof (re));
  re.translate = (unsigned char *) trans;
  s = re_compile_pattern ("\\w", 2, &re);

  if (s != NULL)
    {
      printf ("failed to compile pattern \"\\w\": %s\n", s);
      result = 1;
    }
  else
    {
      int ret = re_search (&re, ".,!abc", 6, 0, 6, NULL);
      if (ret != 3)
	{
	  printf ("3rd re_search returned %d\n", ret);
	  result = 1;
	}

      ret = re_search (&re, "\xae\xf7\xc4\xd6", 4, 0, 4, NULL);
      if (ret != 2)
	{
	  printf ("4th re_search returned %d\n", ret);
	  result = 1;
	}
      re.translate = NULL;
      regfree (&re);
    }

  memset (&re, 0, sizeof (re));
  re.translate = (unsigned char *) trans;
  s = re_compile_pattern ("[[:DIGIT:]]", 11, &re);
  if (s == NULL)
    {
      puts ("compilation of \"[[:DIGIT:]]\" pattern unexpectedly succeeded: "
	    "length 11");
      result = 1;
    }

  memset (&re, 0, sizeof (re));
  re.translate = (unsigned char *) trans;
  s = re_compile_pattern ("[[:DIGIT:]]", 2, &re);
  if (s == NULL)
    {
      puts ("compilation of \"[[:DIGIT:]]\" pattern unexpectedly succeeded: "
	    "length 2");
      result = 1;
    }

  return result;
}
