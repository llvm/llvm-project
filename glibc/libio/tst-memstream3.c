/* Test for open_memstream implementation.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>


#ifndef CHAR_T
# define CHAR_T char
# define W(o) o
# define OPEN_MEMSTREAM open_memstream
# define PRINTF printf
# define FWRITE_FUNC fwrite
# define FPUTC fputc
# define STRCMP strcmp
#endif

#define S(s) S1 (s)
#define S1(s) #s

static void
mcheck_abort (enum mcheck_status ev)
{
  printf ("mecheck failed with status %d\n", (int) ev);
  exit (1);
}

static void
error_printf (int line, const char *fmt, ...)
{
  va_list ap;

  printf ("error: %s:%i: ", __FILE__, line);
  va_start (ap, fmt);
  vprintf (fmt, ap);
  va_end (ap);
}

#define ERROR_RET1(...) \
  { error_printf(__LINE__, __VA_ARGS__); return 1; }

static int
do_test_bz18241 (void)
{
  CHAR_T *buf;
  size_t size;

  FILE *fp = OPEN_MEMSTREAM (&buf, &size);
  if (fp == NULL)
    ERROR_RET1 ("%s failed\n", S(OPEN_MEMSTREAM));

  if (FPUTC (W('a'), fp) != W('a'))
    ERROR_RET1 ("%s failed (errno = %d)\n", S(FPUTC), errno);
  if (fflush (fp) != 0)
    ERROR_RET1 ("fflush failed (errno = %d)\n", errno);
  if (fseek (fp, -2, SEEK_SET) != -1)
    ERROR_RET1 ("fseek failed (errno = %d)\n", errno);
  if (errno != EINVAL)
    ERROR_RET1 ("errno != EINVAL\n");
  if (ftell (fp) != 1)
    ERROR_RET1 ("ftell failed (errno = %d)\n", errno);
  if (ferror (fp) != 0)
    ERROR_RET1 ("ferror != 0\n");

  if (fseek (fp, -1, SEEK_CUR) == -1)
    ERROR_RET1 ("fseek failed (errno = %d)\n", errno);
  if (ftell (fp) != 0)
    ERROR_RET1 ("ftell failed (errno = %d)\n", errno);
  if (ferror (fp) != 0)
    ERROR_RET1 ("ferror != 0\n");
  if (FPUTC (W('b'), fp) != W('b'))
    ERROR_RET1 ("%s failed (errno = %d)\n", S(FPUTC), errno);
  if (fflush (fp) != 0)
    ERROR_RET1 ("fflush failed (errno = %d)\n", errno);

  if (fclose (fp) != 0)
    ERROR_RET1 ("fclose failed (errno = %d\n", errno);

  if (STRCMP (buf, W("b")) != 0)
    ERROR_RET1 ("%s failed\n", S(STRCMP));

  free (buf);

  return 0;
}

static int
do_test_bz20181 (void)
{
  CHAR_T *buf;
  size_t size;
  size_t ret;

  FILE *fp = OPEN_MEMSTREAM (&buf, &size);
  if (fp == NULL)
    ERROR_RET1 ("%s failed\n", S(OPEN_MEMSTREAM));

  if ((ret = FWRITE_FUNC (W("abc"), 1, 3, fp)) != 3)
    ERROR_RET1 ("%s failed (errno = %d)\n", S(FWRITE_FUNC), errno);

  if (fseek (fp, 0, SEEK_SET) != 0)
    ERROR_RET1 ("fseek failed (errno = %d)\n", errno);

  if (FWRITE_FUNC (W("z"), 1, 1, fp) != 1)
    ERROR_RET1 ("%s failed (errno = %d)\n", S(FWRITE_FUNC), errno);

  if (fflush (fp) != 0)
    ERROR_RET1 ("fflush failed (errno = %d)\n", errno);

  /* Avoid truncating the buffer on close.  */
  if (fseek (fp, 3, SEEK_SET) != 0)
    ERROR_RET1 ("fseek failed (errno = %d)\n", errno);

  if (fclose (fp) != 0)
    ERROR_RET1 ("fclose failed (errno = %d\n", errno);

  if (size != 3)
    ERROR_RET1 ("size != 3\n");

  if (buf[0] != W('z')
      || buf[1] != W('b')
      || buf[2] != W('c'))
    {
      PRINTF (W("error: buf {%c,%c,%c} != {z,b,c}\n"),
	      buf[0], buf[1], buf[2]);
      return 1;
    }

  free (buf);

  return 0;
}

static int
do_test (void)
{
  int ret = 0;

  mcheck_pedantic (mcheck_abort);

  ret += do_test_bz18241 ();
  ret += do_test_bz20181 ();

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
