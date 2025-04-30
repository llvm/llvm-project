/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@arthur.rhein-neckar.de>, 1999.

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

#include <grp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static int errors;

static void
write_users (FILE *f, int large_pos, int pos)
{
  int i;

  if (pos == large_pos)
    {
      if (large_pos == 3)
	fprintf (f, ":three");

      /* we need more than 2048 bytes for proper testing.  */
      for (i = 0; i < 500; i++)
	fprintf (f, ",user%03d", i);
    }
  fprintf (f, "\n");

}

static void
write_group (const char *filename, int pos)
{
  FILE *f;

  f = fopen (filename, "w");
  fprintf (f, "one:x:1:one");
  write_users (f, pos, 1);
  fprintf (f, "two:x:2:two");
  write_users (f, pos, 2);
  fprintf (f, "three:x:3");
  write_users (f, pos, 3);
  fclose (f);
}

static void
test_entry (const char *name, gid_t gid, struct group *g)
{
  if (!g)
    {
      printf ("Error: Entry is empty\n");
      errors++;
      return;
    }

  if ((g->gr_gid == gid) && (strcmp (g->gr_name, name) == 0))
    printf ("Ok: %s: %d\n", g->gr_name, g->gr_gid);
  else
    {
      printf ("Error: %s: %d should be: %s: %d\n", g->gr_name, g->gr_gid,
	      name, gid);
      errors++;
    }
}


static void
test_fgetgrent (const char *filename)
{
  struct group *g;
  FILE *f;

  f = fopen (filename,"r");

  g = fgetgrent (f);
  test_entry ("one", 1, g);
  g = fgetgrent (f);
  test_entry ("two", 2, g);
  g = fgetgrent (f);
  test_entry ("three", 3, g);
  fclose (f);
}


int
main (int argc, char *argv[])
{
  char file[] = "/tmp/tst_fgetgrent.XXXXXX";
  int fd = mkstemp (file);
  if (fd == -1)
    {
      printf ("mkstemp failed: %m\n");
      return 1;
    }
  close (fd);
  int i = 0;

  if (argc > 1)
    i = atoi (argv[1]);
  if (i > 3)
    i = 3;
  if (i)
    printf ("Large group is group: %d\n", i);
  else
    printf ("Not using a large group\n");
  write_group (file, i);
  test_fgetgrent (file);

  remove (file);

  return (errors != 0);
}
