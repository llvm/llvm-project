/* fmemopen tests.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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


#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <stdint.h>
#include <errno.h>

/* Check fmemopen with user provided buffer open for write.  */
static int
do_test_with_buffer (void)
{
  int result = 0;
  char buf[100];
  const size_t nbuf = sizeof (buf);

  FILE *fp = fmemopen (buf, nbuf, "w");
  if (fp == NULL)
    {
      printf ("FAIL: fmemopen failed (%s)\n", __FUNCTION__);
      return 1;
    }

  /* Default write operation, check if file position is correct after it.  */
  static const char str[] = "hello world";
  const size_t nstr = sizeof (str) - 1;
  fputs (str, fp);
  off_t o = ftello (fp);
  if (o != nstr)
    {
      printf ("FAIL: first ftello returned %jd, expected %zu\n",
	      (intmax_t)o, nstr);
      result = 1;
    }

  /* Rewind stream and seek tests, the position size should be equal to
     buffer size provided in open function.  */
  rewind (fp);
  o = ftello (fp);
  if (o != 0)
    {
      printf ("FAIL: second ftello returned %jd, expected 0\n",
	      (intmax_t)o);
      result = 1;
    }
  if (fseeko (fp, 0, SEEK_END) != 0)
    {
      printf ("FAIL: fseeko failed\n");
      result = 1;
    }
  o = ftello (fp);
  if (o != nstr)
    {
      printf ("FAIL: third ftello returned %jd, expected %zu\n",
	      (intmax_t)o, nstr);
      result = 1;
    }

  /* Rewind the stream and recheck by using a shorter string.  */
  rewind (fp);
  static const char str2[] = "just hello";
  const size_t nstr2 = sizeof (str2) - 1;
  assert (nstr2 < nstr);
  fputs (str2, fp);
  o = ftello (fp);
  if (o != nstr2)
    {
      printf ("FAIL: fourth ftello returned %jd, expected %zu\n",
	      (intmax_t)o, nstr2);
      result = 1;
    }
  fclose (fp);

  /* Again, but now with a larger string.  */
  static const char str3[] = "just hellod";
  if (strcmp (buf, str3) != 0)
    {
      printf ("FAIL: final string is \"%s\", expected \"%s\"\n",
              buf, str3);
      result = 1;
    }
  return result;
}

/* Check fmemopen without user provided buffer open for write.  */
static int
do_test_without_buffer (void)
{
  int result = 0;
  const size_t nbuf = 100;

  FILE *fp = fmemopen (NULL, nbuf, "w");
  if (fp == NULL)
    {
      printf ("FAIL: fmemopen failed (%s)\n", __FUNCTION__);
      return 1;
    }

  static const char str[] = "hello world";
  const size_t nstr = sizeof (str) - 1;

  /* Default write operation, check if file position is correct after it.  */
  fputs (str, fp);
  off_t o = ftello (fp);
  if (o != nstr)
    {
      printf ("FAIL: first ftello returned %jd, expected %zu\n",
	      (intmax_t) o, nstr);
      result = 1;
    }
  if (fseeko (fp, 0, SEEK_END) != 0)
    {
      printf ("FAIL: fseeko failed\n");
      result = 1;
    }
  o = ftello (fp);
  if (o != nstr)
    {
      printf ("FAIL: second ftello returned %jd, expected %zu\n",
	      (intmax_t) o, nbuf);
      result = 1;
    }

  /* Rewind the stream and recheck by using a shorter string.  */
  rewind (fp);
  static const char str2[] = "just hello";
  const size_t nstr2 = sizeof (str2) - 1;
  assert (nstr2 < nstr);
  fputs (str2, fp);
  o = ftello (fp);
  if (o != nstr2)
    {
      printf ("FAIL: third ftello returned %jd, expected %zu\n",
	      (intmax_t) o, nstr2);
      result = 1;
    }
  fclose (fp);

  return result;
}

/* Check fmemopen with a buffer lenght of zero.  */
static int
do_test_length_zero (void)
{
  int result = 0;
  FILE *fp;
#define BUFCONTENTS "testing buffer"
  char buf[100] = BUFCONTENTS;
  const size_t nbuf = 0;
  int r;

  fp = fmemopen (buf, nbuf, "r");
  if (fp == NULL)
    {
      printf ("FAIL: fmemopen failed (%s)\n", __FUNCTION__);
      return 1;
    }

  /* Reading any data on a zero-length buffer should return EOF.  */
  if ((r = fgetc (fp)) != EOF)
    {
      printf ("FAIL: fgetc on a zero-length returned: %d\n", r);
      result = 1;
    }
  off_t o = ftello (fp);
  if (o != 0)
    {
      printf ("FAIL: first ftello returned %jd, expected 0\n",
	      (intmax_t) o);
      result = 1;
    }
  fclose (fp);

  /* Writing any data shall start at current position and shall not pass
     current buffer size beyond the size in fmemopen call.  */
  fp = fmemopen (buf, nbuf, "w");
  if (fp == NULL)
    {
      printf ("FAIL: second fmemopen failed (%s)\n", __FUNCTION__);
      return 1;
    }

  static const char str[] = "hello world";
  /* Because of buffering, the fputs call itself will not fail. However the
     final buffer should be not changed because length 0 was passed to the
     fmemopen call.  */
  fputs (str, fp);
  r = 0;
  errno = 0;
  if (fflush (fp) != EOF)
    {
      printf ("FAIL: fflush did not return EOF\n");
      fclose (fp);
      return 1;
    }
  if (errno != ENOSPC)
    {
      printf ("FAIL: errno is %i (expected ENOSPC)\n", errno);
      fclose (fp);
      return 1;
    }

  fclose (fp);

  if (strcmp (buf, BUFCONTENTS) != 0)
    {
      printf ("FAIL: strcmp (%s, %s) failed\n", buf, BUFCONTENTS);
      return 1;
    }

  /* Different than 'w' mode, 'w+' truncates the buffer.  */
  fp = fmemopen (buf, nbuf, "w+");
  if (fp == NULL)
    {
      printf ("FAIL: third fmemopen failed (%s)\n", __FUNCTION__);
      return 1;
    }

  fclose (fp);

  if (strcmp (buf, "") != 0)
    {
      printf ("FAIL: strcmp (%s, \"\") failed\n", buf);
      return 1;
    }

  return result;
}

static int
do_test (void)
{
  int ret = 0;

  ret += do_test_with_buffer ();
  ret += do_test_without_buffer ();
  ret += do_test_length_zero ();

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
