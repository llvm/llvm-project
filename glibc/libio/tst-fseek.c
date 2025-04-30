/* Verify that fseek/ftell combination works for wide chars.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <errno.h>
#include <wchar.h>
#include <unistd.h>
#include <string.h>

#include <support/temp_file.h>

static int
do_seek_end (FILE *fp)
{
  long save;

  if (fputws (L"abc\n", fp) == -1)
    {
      printf ("do_seek_end: fputws: %s\n", strerror (errno));
      return 1;
    }

  save = ftell (fp);
  rewind (fp);

  if (fseek (fp, 0, SEEK_END) == -1)
    {
      printf ("do_seek_end: fseek: %s\n", strerror (errno));
      return 1;
    }

  if (save != ftell (fp))
    {
      printf ("save = %ld, ftell = %ld\n", save, ftell (fp));
      return 1;
    }

  return 0;
}

int
do_seek_set (FILE *fp)
{
  long save1, save2;

  if (fputws (L"ゅう\n", fp) == -1)
    {
      printf ("seek_set: fputws(1): %s\n", strerror (errno));
      return 1;
    }

  save1 = ftell (fp);

  if (fputws (L"ゅう\n", fp) == -1)
    {
      printf ("seek_set: fputws(2): %s\n", strerror (errno));
      return 1;
    }

  save2 = ftell (fp);

  if (fputws (L"ゅう\n", fp) == -1)
    {
      printf ("seek_set: fputws(3): %s\n", strerror (errno));
      return 1;
    }

  if (fseek (fp, save1, SEEK_SET) == -1)
    {
      printf ("seek_set: fseek(1): %s\n", strerror (errno));
      return 1;
    }

  if (save1 != ftell (fp))
    {
      printf ("save1 = %ld, ftell = %ld\n", save1, ftell (fp));
      return 1;
    }

  if (fseek (fp, save2, SEEK_SET) == -1)
    {
      printf ("seek_set: fseek(2): %s\n", strerror (errno));
      return 1;
    }

  if (save2 != ftell (fp))
    {
      printf ("save2 = %ld, ftell = %ld\n", save2, ftell (fp));
      return 1;
    }

  return 0;
}

static int
do_test (void)
{
  if (setlocale (LC_ALL, "ja_JP.UTF-8") == NULL)
    {
      printf ("Cannot set ja_JP.UTF-8 locale.\n");
      exit (1);
    }

  /* Retain messages in English.  */
  if (setlocale (LC_MESSAGES, "en_US.ISO-8859-1") == NULL)
    {
      printf ("Cannot set LC_MESSAGES to en_US.ISO-8859-1 locale.\n");
      exit (1);
    }

  int ret = 0;
  char *filename;
  int fd = create_temp_file ("tst-fseek.out", &filename);

  if (fd == -1)
    return 1;

  FILE *fp = fdopen (fd, "w+");
  if (fp == NULL)
    {
      printf ("seek_set: fopen: %s\n", strerror (errno));
      close (fd);
      return 1;
    }

  if (do_seek_set (fp))
    {
      printf ("SEEK_SET test failed\n");
      ret = 1;
    }

  /* Reopen the file.  */
  fclose (fp);
  fp = fopen (filename, "w+");
  if (fp == NULL)
    {
      printf ("seek_end: fopen: %s\n", strerror (errno));
      return 1;
    }

  if (do_seek_end (fp))
    {
      printf ("SEEK_END test failed\n");
      ret = 1;
    }

  fclose (fp);

  return ret;
}

#include <support/test-driver.c>
