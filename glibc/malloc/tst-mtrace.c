/* Test program for mtrace.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <mcheck.h>
#include <paths.h>
#include <search.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void print (const void *node, VISIT value, int level);

/* Used for several purposes.  */
static FILE *fp;


static int
do_test (void)
{
  void *root = NULL;
  size_t linelen = 0;
  char *line = NULL;

  /* Enable memory usage tracing.  */
  mtrace ();

  /* Perform some operations which definitely will allocate some
     memory.  */
  fp = fopen (__FILE__, "r");
  if (fp == NULL)
    /* Shouldn't happen since this program is executed in the source
       directory.  */
    abort ();

  while (!feof (fp))
    {
      char **p;
      char *copy;
      ssize_t n = getline (&line, &linelen, fp);

      if (n < 0)
        break;

      if (n == 0)
        continue;

      copy = strdup (line);
      if (copy == NULL)
        abort ();

      p = (char **) tsearch (copy, &root,
                             (int (*)(const void *, const void *))strcmp);
      if (*p != copy)
        /* This line wasn't added.  */
        free (copy);
    }

  fclose (fp);

  fp = fopen (_PATH_DEVNULL, "w");
  if (fp != NULL)
    {
      /* Write something through stdout.  */
      twalk (root, print);

      fclose (fp);
    }

  /* Free everything.  */
  tdestroy (root, free);

  /* Also the line buffer.  */
  free (line);

  /* That's it.  */
  return 0;
}


static void
print (const void *node, VISIT value, int level)
{
  static int cnt;
  if (value == postorder || value == leaf)
    fprintf (fp, "%3d: %s", ++cnt, *(const char **) node);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
