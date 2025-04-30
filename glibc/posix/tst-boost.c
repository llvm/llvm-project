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

void
frob_escapes (char *src, int pattern)
{
  char *dst;

  for (dst = src; *src != '\0'; dst++, src++)
    {
      if (*src == '\\')
	{
	  switch (src[1])
	    {
	    case 't':
	      src++;
	      *dst = '\t';
	      continue;
	    case 'n':
	      src++;
	      *dst = '\n';
	      continue;
	    case 'r':
	      src++;
	      *dst = '\r';
	      continue;
	    case '\\':
	    case '^':
	    case '{':
	    case '|':
	    case '}':
	      if (!pattern)
		{
		  src++;
		  *dst = *src;
		  continue;
		}
	      break;
	    }
	}
      if (src != dst)
	*dst = *src;
    }
  *dst = '\0';
}

int
main (int argc, char **argv)
{
  int ret = 0, n;
  char *line = NULL;
  size_t line_len = 0;
  ssize_t len;
  FILE *f;
  char *pattern, *string;
  int flags = REG_EXTENDED;
  int eflags = 0;
  regex_t re;
  regmatch_t rm[20];

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

  while ((len = getline (&line, &line_len, f)) > 0)
    {
      char *p, *q;
      int i;

      if (line[len - 1] == '\n')
	line[--len] = '\0';

      puts (line);

      if (line[0] == ';')
	continue;

      if (line[0] == '\0')
	continue;

      if (line[0] == '-')
	{
	  if (strstr (line, "REG_BASIC"))
	    flags = 0;
	  else
	    flags = REG_EXTENDED;
	  if (strstr (line, "REG_ICASE"))
	    flags |= REG_ICASE;
	  if (strstr (line, "REG_NEWLINE"))
	    flags |= REG_NEWLINE;
	  eflags = 0;
	  if (strstr (line, "REG_NOTBOL"))
	    eflags |= REG_NOTBOL;
	  if (strstr (line, "REG_NOTEOL"))
	    eflags |= REG_NOTEOL;
	  continue;
	}

      pattern = line + strspn (line, " \t");
      if (*pattern == '\0')
	continue;
      p = pattern + strcspn (pattern, " \t");
      if (*p == '\0')
	continue;
      *p++ = '\0';

      string = p + strspn (p, " \t");
      if (*string == '\0')
	continue;
      if (*string == '"')
	{
	  string++;
	  p = strchr (string, '"');
	  if (p == NULL)
	    continue;
	  *p++ = '\0';
	}
      else
	{
	  p = string + strcspn (string, " \t");
	  if (*string == '!')
	    string = NULL;
	  else if (*p == '\0')
	    continue;
	  else
	    *p++ = '\0';
	}

      frob_escapes (pattern, 1);
      if (string != NULL)
	frob_escapes (string, 0);

      n = regcomp (&re, pattern, flags);
      if (n != 0)
	{
	  if (string != NULL)
	    {
	      char buf[500];
	      regerror (n, &re, buf, sizeof (buf));
	      printf ("FAIL regcomp unexpectedly failed: %s\n",
		      buf);
	      ret = 1;
	    }
	  continue;
	}
      else if (string == NULL)
	{
	  regfree (&re);
	  puts ("FAIL regcomp unpexpectedly succeeded");
	  ret = 1;
	  continue;
	}

      if (regexec (&re, string, 20, rm, eflags))
	{
	  for (i = 0; i < 20; ++i)
	    {
	      rm[i].rm_so = -1;
	      rm[i].rm_eo = -1;
	    }
	}

      regfree (&re);

      for (i = 0; i < 20 && *p != '\0'; ++i)
	{
	  int rm_so, rm_eo;

	  rm_so = strtol (p, &q, 10);
	  if (p == q)
	    break;
	  p = q;

	  rm_eo = strtol (p, &q, 10);
	  if (p == q)
	    break;
	  p = q;

	  if (rm[i].rm_so != rm_so || rm[i].rm_eo != rm_eo)
	    {
	      printf ("FAIL rm[%d] %d..%d != expected %d..%d\n",
		      i, rm[i].rm_so, rm[i].rm_eo, rm_so, rm_eo);
	      ret = 1;
	      break;
	    }
	}
    }

  free (line);
  fclose (f);
  return ret;
}
