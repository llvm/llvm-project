/* Regular expression tests.
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

int
main (int argc, char **argv)
{
  int ret = 0;
  char *line = NULL;
  size_t line_len = 0;
  ssize_t len;
  FILE *f;
  char *pattern = NULL, *string = NULL;
  regmatch_t rm[20];
  size_t pattern_alloced = 0, string_alloced = 0;
  int ignorecase = 0;
  int pattern_valid = 0, rm_valid = 0;
  size_t linenum;

  mtrace ();

  if (argc < 2)
    {
      fprintf (stderr, "Missing test filename\n");
      return 1;
    }

  f = fopen (argv[1], "r");
  if (f == NULL)
    {
      fprintf (stderr, "Couldn't open %s\n", argv[1]);
      return 1;
    }

  if ((len = getline (&line, &line_len, f)) <= 0
      || strncmp (line, "# PCRE", 6) != 0)
    {
      fprintf (stderr, "Not a PCRE test file\n");
      fclose (f);
      free (line);
      return 1;
    }

  linenum = 1;

  while ((len = getline (&line, &line_len, f)) > 0)
    {
      char *p;
      unsigned long num;

      ++linenum;

      if (line[len - 1] == '\n')
	line[--len] = '\0';

      if (line[0] == '#')
	continue;

      if (line[0] == '\0')
	{
	  /* End of test.  */
	  ignorecase = 0;
	  pattern_valid = 0;
	  rm_valid = 0;
	  continue;
	}

      if (line[0] == '/')
	{
	  /* Pattern.  */
	  p = strrchr (line + 1, '/');

	  pattern_valid = 0;
	  rm_valid = 0;
	  if (p == NULL)
	    {
	      printf ("%zd: Invalid pattern line: %s\n", linenum, line);
	      ret = 1;
	      continue;
	    }

	  if (p[1] == 'i' && p[2] == '\0')
	    ignorecase = 1;
	  else if (p[1] != '\0')
	    {
	      printf ("%zd: Invalid pattern line: %s\n", linenum, line);
	      ret = 1;
	      continue;
	    }

	  if (pattern_alloced < (size_t) (p - line))
	    {
	      pattern = realloc (pattern, p - line);
	      if (pattern == NULL)
		{
		  printf ("%zd: Cannot record pattern: %m\n", linenum);
		  ret = 1;
		  break;
		}
	      pattern_alloced = p - line;
	    }

	  memcpy (pattern, line + 1, p - line - 1);
	  pattern[p - line - 1] = '\0';
	  pattern_valid = 1;
	  continue;
	}

      if (strncmp (line, "    ", 4) == 0)
	{
	  regex_t re;
	  int n;

	  if (!pattern_valid)
	    {
	      printf ("%zd: No previous valid pattern %s\n", linenum, line);
	      continue;
	    }

	  if (string_alloced < (size_t) (len - 3))
	    {
	      string = realloc (string, len - 3);
	      if (string == NULL)
		{
		  printf ("%zd: Cannot record search string: %m\n", linenum);
		  ret = 1;
		  break;
		}
	      string_alloced = len - 3;
	    }

	  memcpy (string, line + 4, len - 3);

	  n = regcomp (&re, pattern,
		       REG_EXTENDED | (ignorecase ? REG_ICASE : 0));
	  if (n != 0)
	    {
	      char buf[500];
	      regerror (n, &re, buf, sizeof (buf));
	      printf ("%zd: regcomp failed for %s: %s\n",
		      linenum, pattern, buf);
	      ret = 1;
	      continue;
	    }

	  if (regexec (&re, string, 20, rm, 0))
	    {
	      rm[0].rm_so = -1;
	      rm[0].rm_eo = -1;
	    }

	  regfree (&re);
	  rm_valid = 1;
	  continue;
	}

      if (!rm_valid)
	{
	  printf ("%zd: No preceeding pattern or search string\n", linenum);
	  ret = 1;
	  continue;
	}

      if (strcmp (line, "No match") == 0)
	{
	  if (rm[0].rm_so != -1 || rm[0].rm_eo != -1)
	    {
	      printf ("%zd: /%s/ on %s unexpectedly matched %d..%d\n",
		      linenum, pattern, string, rm[0].rm_so, rm[0].rm_eo);
	      ret = 1;
	    }

	  continue;
	}

      p = line;
      if (*p == ' ')
        ++p;

      num = strtoul (p, &p, 10);
      if (num >= 20 || *p != ':' || p[1] != ' ')
	{
	  printf ("%zd: Invalid line %s\n", linenum, line);
	  ret = 1;
	  continue;
	}

      if (rm[num].rm_so == -1 || rm[num].rm_eo == -1)
	{
	  if (strcmp (p + 2, "<unset>") != 0)
	    {
	      printf ("%zd: /%s/ on %s unexpectedly failed to match register %ld %d..%d\n",
		      linenum, pattern, string, num,
		      rm[num].rm_so, rm[num].rm_eo);
	      ret = 1;
	    }
	  continue;
	}

      if (rm[num].rm_eo < rm[num].rm_so
	  || rm[num].rm_eo - rm[num].rm_so != len - (p + 2 - line)
	  || strncmp (p + 2, string + rm[num].rm_so,
		      rm[num].rm_eo - rm[num].rm_so) != 0)
	{
	  printf ("%zd: /%s/ on %s unexpectedly failed to match %s for register %ld %d..%d\n",
		  linenum, pattern, string, p + 2, num,
		  rm[num].rm_so, rm[num].rm_eo);
	  ret = 1;
	  continue;
	}
    }

  free (pattern);
  free (string);
  free (line);
  fclose (f);
  return ret;
}
