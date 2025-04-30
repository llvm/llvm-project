/* Test that assigning to stdin redirects input (bug 24153).
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* Prevent getchar -> getc inline expansion.  */
#define __NO_INLINE__
#pragma GCC optimize ("O0")

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/temp_file.h>
#include <support/xstdio.h>
#include <support/xunistd.h>
#include <wchar.h>

static int
call_vscanf (const char *format, ...)
{
  va_list ap;
  va_start (ap, format);
  int ret = vscanf (format, ap);
  va_end (ap);
  return ret;
}

static int
call_vwscanf (const wchar_t *format, ...)
{
  va_list ap;
  va_start (ap, format);
  int ret = vwscanf (format, ap);
  va_end (ap);
  return ret;
}

static void
narrow (const char *path)
{
  FILE *old_stdin = stdin;
  stdin = xfopen (path, "r");

  TEST_COMPARE (getchar (), 'a');
  TEST_COMPARE (getchar_unlocked (), 'b');
  char ch = 1;
  TEST_COMPARE (scanf ("%c", &ch), 1);
  TEST_COMPARE (ch, 'c');
  TEST_COMPARE (call_vscanf ("%c", &ch), 1);
  TEST_COMPARE (ch, 'd');
  char buf[8];
  memset (buf, 'X', sizeof (buf));

  /* Legacy interface.  */
  extern char *gets (char *);
  TEST_VERIFY (gets (buf) == buf);
  TEST_COMPARE_BLOB (buf, sizeof (buf), "ef\0XXXXX", sizeof (buf));

  fclose (stdin);
  stdin = old_stdin;
}

static void
wide (const char *path)
{
  FILE *old_stdin = stdin;
  stdin = xfopen (path, "r");

  TEST_COMPARE (getwchar (), L'a');
  TEST_COMPARE (getwchar_unlocked (), L'b');
  wchar_t ch = 1;
  TEST_COMPARE (wscanf (L"%lc", &ch), 1);
  TEST_COMPARE (ch, L'c');
  TEST_COMPARE (call_vwscanf (L"%lc", &ch), 1);
  TEST_COMPARE (ch, L'd');

  fclose (stdin);
  stdin = old_stdin;
}

static int
do_test (void)
{
  char *path;
  {
    int fd = create_temp_file ("tst-bz24153-", &path);
    TEST_VERIFY_EXIT (fd >= 0);
    xwrite (fd, "abcdef", strlen ("abcdef"));
    xclose (fd);
  }

  narrow (path);
  wide (path);

  free (path);
  return 0;
}

#include <support/test-driver.c>
