/* Test program for ungetc/fseekpos interaction.
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

#include <stdio.h>

static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static const char pattern[] = "abcdefghijklmnopqrstuvwxyz";
static char *temp_file;

static void
do_prepare (void)
{
  int fd = create_temp_file ("bug-ungetc.", &temp_file);
  if (fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }
  write (fd, pattern, sizeof (pattern) - 1);
  close (fd);
}

#define TEST_POS 5

static int
do_one_test (int mode)
{
  FILE *f = fopen (temp_file, "r");
  if (f == NULL)
    {
      printf ("%d: could not open temporary file: %m\n", mode);
      return 1;
    }

  int i;
  for (i = 0; i < TEST_POS; ++i)
    getc (f);

  fpos_t p;
  if (fgetpos (f, &p) != 0)
    {
      printf ("%d: fgetpos failed: %m\n", mode);
      return 1;
    }

  if (mode)
    {
      if (fseek (f, 0, SEEK_SET) != 0)
        {
          printf ("%d: fseek failed: %m\n", mode);
          return 1;
        }

      for (i = 0; i < TEST_POS - (mode >= 2); ++i)
        getc (f);
    }

  if (mode != 2 && ungetc ('X', f) != 'X')
    {
      printf ("%d: ungetc failed\n", mode);
      return 1;
    }

  if (mode == 3 && getc (f) != 'X')
    {
      printf ("%d: getc after ungetc did not return X\n", mode);
      return 1;
    }

  if (fsetpos (f, &p) != 0)
    {
      printf ("%d: fsetpos failed: %m\n", mode);
      return 1;
    }

  if (getc (f) != pattern[TEST_POS])
    {
      printf ("%d: getc did not return %c\n", mode, pattern[TEST_POS]);
      return 1;
    }

  fclose (f);

  return 0;
}

static int
do_test (void)
{
  int mode, ret = 0;
  for (mode = 0; mode <= 4; mode++)
    ret |= do_one_test (mode);
  return ret;
}
