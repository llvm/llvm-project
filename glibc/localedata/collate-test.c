/* Test collation function using real data.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <error.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct lines
{
  char *key;
  char *line;
};

static int xstrcoll (const void *, const void *);

int
main (int argc, char *argv[])
{
  int result = 0;
  size_t nstrings, nstrings_max;
  struct lines *strings;
  char *line = NULL;
  size_t len = 0;
  size_t n;

  if (argc < 2)
    error (1, 0, "usage: %s <random seed>", argv[0]);

  setlocale (LC_ALL, "");

  nstrings_max = 100;
  nstrings = 0;
  strings = (struct lines *) malloc (nstrings_max * sizeof (struct lines));
  if (strings == NULL)
    {
      perror (argv[0]);
      exit (1);
    }

  while (1)
    {
      int l;
      if (getline (&line, &len, stdin) < 0)
	break;

      if (nstrings == nstrings_max)
	{
	  strings = (struct lines *) realloc (strings,
					      (nstrings_max *= 2)
					       * sizeof (*strings));
	  if (strings == NULL)
	    {
	      perror (argv[0]);
	      exit (1);
	    }
	}
      strings[nstrings].line = strdup (line);
      l = strcspn (line, ":(;");
      while (l > 0 && isspace (line[l - 1]))
	--l;
      strings[nstrings].key = strndup (line, l);
      ++nstrings;
    }
  free (line);

  /* First shuffle.  */
  srandom (atoi (argv[1]));
  for (n = 0; n < 10 * nstrings; ++n)
    {
      int r1, r2, r;
      size_t idx1 = random () % nstrings;
      size_t idx2 = random () % nstrings;
      struct lines tmp = strings[idx1];
      strings[idx1] = strings[idx2];
      strings[idx2] = tmp;

      /* While we are at it a first little test.  */
      r1 = strcoll (strings[idx1].key, strings[idx2].key);
      r2 = strcoll (strings[idx2].key, strings[idx1].key);
      r = r1 * r2;

      if (r > 0 || (r == 0 && r1 != 0) || (r == 0 && r2 != 0))
	printf ("`%s' and `%s' collate wrong: %d vs. %d\n",
		strings[idx1].key, strings[idx2].key, r1, r2);
    }

  /* Now sort.  */
  qsort (strings, nstrings, sizeof (struct lines), xstrcoll);

  /* Print the result.  */
  for (n = 0; n < nstrings; ++n)
    {
      fputs (strings[n].line, stdout);
      free (strings[n].line);
      free (strings[n].key);
    }
  free (strings);

  return result;
}


static int
xstrcoll (const void *ptr1, const void *ptr2)
{
  const struct lines *l1 = (const struct lines *) ptr1;
  const struct lines *l2 = (const struct lines *) ptr2;

  return strcoll (l1->key, l2->key);
}
