/* Copyright (C) 2008-2021 Free Software Foundation, Inc.
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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#if __GLIBC_USE_DEPRECATED_SCANF
# error "This file should not be compiled with deprecated scanf"
#endif

#define FAIL() \
  do {							\
    result = 1;						\
    printf ("test at line %d failed\n", __LINE__);	\
  } while (0)

static int __attribute__ ((format (scanf, 2, 3)))
xsscanf (const char *str, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  int ret = vsscanf (str, fmt, ap);
  va_end (ap);
  return ret;
}

static int __attribute__ ((format (scanf, 1, 2)))
xscanf (const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  int ret = vscanf (fmt, ap);
  va_end (ap);
  return ret;
}

static int __attribute__ ((format (scanf, 2, 3)))
xfscanf (FILE *f, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  int ret = vfscanf (f, fmt, ap);
  va_end (ap);
  return ret;
}

int
main (void)
{
  wchar_t *lsp;
  char *sp;
  float f;
  double d;
  char c[8];
  int result = 0;

  if (xsscanf (" 0.25s x", "%e%3c", &f, c) != 2)
    FAIL ();
  else if (f != 0.25 || memcmp (c, "s x", 3) != 0)
    FAIL ();
  if (xsscanf (" 1.25s x", "%ms%2c", &sp, c) != 2)
    FAIL ();
  else
    {
      if (strcmp (sp, "1.25s") != 0 || memcmp (c, " x", 2) != 0)
	FAIL ();
      memset (sp, 'x', sizeof "1.25s");
      free (sp);
    }
  if (xsscanf (" 2.25s x", "%las%2c", &d, c) != 2)
    FAIL ();
  else if (d != 2.25 || memcmp (c, " x", 2) != 0)
    FAIL ();
  if (xsscanf (" 3.25S x", "%4mS%3c", &lsp, c) != 2)
    FAIL ();
  else
    {
      if (wcscmp (lsp, L"3.25") != 0 || memcmp (c, "S x", 3) != 0)
	FAIL ();
      memset (lsp, 'x', sizeof L"3.25");
      free (lsp);
    }
  if (xsscanf ("4.25[0-9.] x", "%m[0-9.]%8c", &sp, c) != 2)
    FAIL ();
  else
    {
      if (strcmp (sp, "4.25") != 0 || memcmp (c, "[0-9.] x", 8) != 0)
	FAIL ();
      memset (sp, 'x', sizeof "4.25");
      free (sp);
    }
  if (xsscanf ("5.25[0-9.] x", "%la[0-9.]%2c", &d, c) != 2)
    FAIL ();
  else if (d != 5.25 || memcmp (c, " x", 2) != 0)
    FAIL ();

  const char *tmpdir = getenv ("TMPDIR");
  if (tmpdir == NULL || tmpdir[0] == '\0')
    tmpdir = "/tmp";

  char fname[strlen (tmpdir) + sizeof "/tst-scanf16.XXXXXX"];
  sprintf (fname, "%s/tst-scanf16.XXXXXX", tmpdir);

  /* Create a temporary file.   */
  int fd = mkstemp (fname);
  if (fd == -1)
    FAIL ();

  FILE *fp = fdopen (fd, "w+");
  if (fp == NULL)
    FAIL ();
  else
    {
      if (fputs (" 1.25s x", fp) == EOF)
	FAIL ();
      if (fseek (fp, 0, SEEK_SET) != 0)
	FAIL ();
      if (xfscanf (fp, "%ms%2c", &sp, c) != 2)
	FAIL ();
      else
	{
	  if (strcmp (sp, "1.25s") != 0 || memcmp (c, " x", 2) != 0)
	    FAIL ();
	  memset (sp, 'x', sizeof "1.25s");
	  free (sp);
	}

      if (freopen (fname, "r", stdin) == NULL)
	FAIL ();
      else
	{
	  if (xscanf ("%ms%2c", &sp, c) != 2)
	    FAIL ();
	  else
	    {
	      if (strcmp (sp, "1.25s") != 0 || memcmp (c, " x", 2) != 0)
		FAIL ();
	      memset (sp, 'x', sizeof "1.25s");
	      free (sp);
	    }
	}

      fclose (fp);
    }

  remove (fname);

  return result;
}
