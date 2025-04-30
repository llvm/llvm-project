/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <glob.h>

int
main (int argc, char *argv[])
{
  int i, j;
  int glob_flags = 0;
  glob_t g;
  int quotes = 1;

  g.gl_offs = 0;

  while ((i = getopt (argc, argv, "bcdeEgmopqstT")) != -1)
    switch(i)
      {
      case 'b':
	glob_flags |= GLOB_BRACE;
	break;
      case 'c':
	glob_flags |= GLOB_NOCHECK;
	break;
      case 'd':
	glob_flags |= GLOB_ONLYDIR;
	break;
      case 'e':
	glob_flags |= GLOB_NOESCAPE;
	break;
      case 'E':
	glob_flags |= GLOB_ERR;
	break;
      case 'g':
	glob_flags |= GLOB_NOMAGIC;
	break;
      case 'm':
	glob_flags |= GLOB_MARK;
	break;
      case 'o':
	glob_flags |= GLOB_DOOFFS;
	g.gl_offs = 1;
	break;
      case 'p':
	glob_flags |= GLOB_PERIOD;
	break;
      case 'q':
	quotes = 0;
	break;
      case 's':
	glob_flags |= GLOB_NOSORT;
	break;
      case 't':
	glob_flags |= GLOB_TILDE;
	break;
      case 'T':
	glob_flags |= GLOB_TILDE_CHECK;
	break;
      default:
	exit (-1);
      }

  if (optind >= argc || chdir (argv[optind]))
    exit(1);

  j = optind + 1;
  if (optind + 1 >= argc)
    exit (1);

  /* Do a glob on each remaining argument.  */
  for (j = optind + 1; j < argc; j++) {
    i = glob (argv[j], glob_flags, NULL, &g);
    if (i != 0)
      break;
    glob_flags |= GLOB_APPEND;
  }

  /* Was there an error? */
  if (i == GLOB_NOSPACE)
    puts ("GLOB_NOSPACE");
  else if (i == GLOB_ABORTED)
    puts ("GLOB_ABORTED");
  else if (i == GLOB_NOMATCH)
    puts ("GLOB_NOMATCH");

  /* If we set an offset, fill in the first field.
     (Unless glob() has filled it in already - which is an error) */
  if ((glob_flags & GLOB_DOOFFS) && g.gl_pathv[0] == NULL)
    g.gl_pathv[0] = (char *) "abc";

  /* Print out the names.  Unless otherwise specified, qoute them.  */
  if (g.gl_pathv)
    {
      for (i = 0; i < g.gl_offs + g.gl_pathc; ++i)
        printf ("%s%s%s\n", quotes ? "`" : "",
		g.gl_pathv[i] ? g.gl_pathv[i] : "(null)",
		quotes ? "'" : "");
    }
  return 0;
}
