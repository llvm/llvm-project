/* Verify that ftell returns the correct value after a read and a write on a
   file opened in a+ mode.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <locale.h>
#include <wchar.h>

/* data points to either char_data or wide_data, depending on whether we're
   testing regular file mode or wide mode respectively.  Similarly,
   fputs_func points to either fputs or fputws.  data_len keeps track of the
   length of the current data and file_len maintains the current file
   length.  */
#define BUF_LEN 4
static void *buf;
static char char_buf[BUF_LEN];
static wchar_t wide_buf[BUF_LEN];
static const void *data;
static const char *char_data = "abcdefghijklmnopqrstuvwxyz";
static const wchar_t *wide_data = L"abcdefghijklmnopqrstuvwxyz";
static size_t data_len;
static size_t file_len;

typedef int (*fputs_func_t) (const void *data, FILE *fp);
fputs_func_t fputs_func;

typedef void *(*fgets_func_t) (void *s, int size, FILE *stream);
fgets_func_t fgets_func;

static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static FILE *
init_file (const char *filename)
{
  FILE *fp = fopen (filename, "w");
  if (fp == NULL)
    {
      printf ("fopen: %m\n");
      return NULL;
    }

  int written = fputs_func (data, fp);

  if (written == EOF)
    {
      printf ("fputs failed to write data\n");
      fclose (fp);
      return NULL;
    }

  file_len = data_len;

  fclose (fp);

  fp = fopen (filename, "a+");
  if (fp == NULL)
    {
      printf ("fopen(a+): %m\n");
      return NULL;
    }

  return fp;
}

static int
do_one_test (const char *filename)
{
  FILE *fp = init_file (filename);

  if (fp == NULL)
    return 1;

  void *ret = fgets_func (buf, BUF_LEN, fp);

  if (ret == NULL)
    {
      printf ("read failed: %m\n");
      fclose (fp);
      return 1;
    }

  int written = fputs_func (data, fp);

  if (written == EOF)
    {
      printf ("fputs failed to write data\n");
      fclose (fp);
      return 1;
    }

  file_len += data_len;

  long off = ftell (fp);

  if (off != file_len)
    {
      printf ("Incorrect offset %ld, expected %zu\n", off, file_len);
      fclose (fp);
      return 1;
    }
  else
    printf ("Correct offset %ld after write.\n", off);

  return 0;
}

/* Run the tests for regular files and wide mode files.  */
static int
do_test (void)
{
  int ret = 0;
  char *filename;
  int fd = create_temp_file ("tst-ftell-append-tmp.", &filename);

  if (fd == -1)
    {
      printf ("create_temp_file: %m\n");
      return 1;
    }

  close (fd);

  /* Tests for regular files.  */
  puts ("Regular mode:");
  fputs_func = (fputs_func_t) fputs;
  fgets_func = (fgets_func_t) fgets;
  data = char_data;
  buf = char_buf;
  data_len = strlen (char_data);
  ret |= do_one_test (filename);

  /* Tests for wide files.  */
  puts ("Wide mode:");
  if (setlocale (LC_ALL, "en_US.UTF-8") == NULL)
    {
      printf ("Cannot set en_US.UTF-8 locale.\n");
      return 1;
    }
  fputs_func = (fputs_func_t) fputws;
  fgets_func = (fgets_func_t) fgetws;
  data = wide_data;
  buf = wide_buf;
  data_len = wcslen (wide_data);
  ret |= do_one_test (filename);

  return ret;
}
