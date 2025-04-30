/* Verify that ftell does not go into an infinite loop when a conversion fails
   due to insufficient space in the buffer.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <errno.h>
#include <unistd.h>

static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

/* Arbitrary number large enough so that the target buffer during conversion is
   not large enough.  */
#define STRING_SIZE (1400)
#define NSTRINGS (2)

static int
do_test (void)
{
  FILE *fp = NULL;
  wchar_t *inputs[NSTRINGS] = {NULL};
  int ret = 1;

  if (setlocale (LC_ALL, "en_US.UTF-8") == NULL)
    {
      printf ("Cannot set en_US.UTF-8 locale.\n");
      goto out;
    }


  /* Generate input from one character, chosen because it has an odd number of
     bytes in UTF-8, making it easier to reproduce the problem:

     NAME    Hiragana letter GO
     CHAR    ご
     UTF-8   E38194
     UCS     3054
     MARC-8  692434  */
  wchar_t seed = L'ご';
  for (int i = 0; i < NSTRINGS; i++)
    {
      inputs[i] = malloc (STRING_SIZE * sizeof (wchar_t));
      if (inputs[i] == NULL)
	{
	  printf ("Failed to allocate memory for inputs: %m\n");
	  goto out;
	}
      wmemset (inputs[i], seed, STRING_SIZE - 1);
      inputs[i][STRING_SIZE - 1] = L'\0';
    }

  char *filename;
  int fd = create_temp_file ("tst-fseek-wide-partial.out", &filename);

  if (fd == -1)
    {
      printf ("create_temp_file: %m\n");
      goto out;
    }

  fp = fdopen (fd, "w+");
  if (fp == NULL)
    {
      printf ("fopen: %m\n");
      close (fd);
      goto out;
    }

  for (int i = 0; i < NSTRINGS; i++)
    {
      printf ("offset: %ld\n", ftell (fp));
      if (fputws (inputs[i], fp) == -1)
	{
	  perror ("fputws");
	  goto out;
	}
    }
  ret = 0;

out:
  if (fp != NULL)
    fclose (fp);
  for (int i = 0; i < NSTRINGS; i++)
    free (inputs[i]);

  return ret;
}
