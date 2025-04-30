/* Test for some *_unlocked stdio interfaces.
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libc-diag.h>

int fd;
static void do_prepare (void);
static int do_test (void);
#define PREPARE(argc, argv) do_prepare ()
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static int
do_test (void)
{
  const char blah[] = "BLAH";
  char buf[strlen (blah) + 1];
  FILE *fp, *f;
  const char *cp;
  char *wp;

  if ((fp = fdopen (fd, "w+")) == NULL)
    exit (1);

  flockfile (fp);

  f = fp;
  cp = blah;
  /* These tests deliberately use fwrite_unlocked with the size
     argument specified as 0, which results in "division by zero"
     warnings from the expansion of that macro (in code that is not
     evaluated for a size of 0).  This applies to the tests of
     fread_unlocked below as well.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdiv-by-zero");
  if (ftello (fp) != 0
      || fwrite_unlocked (blah, blah - blah, strlen (blah), f++) != 0
      || f != fp + 1
      || fwrite_unlocked ("", 5.0, 0, --f) != 0
      || f != fp
      || fwrite_unlocked (cp++, 16, 0.25, fp) != 0
      || cp != blah + 1
      || fwrite_unlocked (--cp, 0.25, 16, fp) != 0
      || cp != blah
      || fwrite_unlocked (blah, 0, -0.0, fp) != 0
      || ftello (fp) != 0)
    {
      puts ("One of fwrite_unlocked tests failed");
      exit (1);
    }
  DIAG_POP_NEEDS_COMMENT;

  if (fwrite_unlocked (blah, 1, strlen (blah) - 2, fp) != strlen (blah) - 2)
    {
      puts ("Could not write string into file");
      exit (1);
    }

  if (putc_unlocked ('A' + 0x1000000, fp) != 'A')
    {
      puts ("putc_unlocked failed");
      exit (1);
    }

  f = fp;
  cp = blah + strlen (blah) - 1;
  if (putc_unlocked (*cp++, f++) != 'H'
      || f != fp + 1
      || cp != strchr (blah, '\0'))
    {
      puts ("fputc_unlocked failed");
      exit (1);
    }

  if (ftello (fp) != (off_t) strlen (blah))
    {
      printf ("Failed to write %zd bytes to temporary file", strlen (blah));
      exit (1);
    }

  rewind (fp);

  f = fp;
  wp = buf;
  memset (buf, ' ', sizeof (buf));
  /* See explanation above.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdiv-by-zero");
  if (ftello (fp) != 0
      || fread_unlocked (buf, buf - buf, strlen (blah), f++) != 0
      || f != fp + 1
      || fread_unlocked (buf, 5.0, 0, --f) != 0
      || f != fp
      || fread_unlocked (wp++, 16, 0.25, fp) != 0
      || wp != buf + 1
      || fread_unlocked (--wp, 0.25, 16, fp) != 0
      || wp != buf
      || fread_unlocked (buf, 0, -0.0, fp) != 0
      || ftello (fp) != 0
      || memcmp (buf, "     ", sizeof (buf)) != 0)
    {
      puts ("One of fread_unlocked tests failed");
      exit (1);
    }
  DIAG_POP_NEEDS_COMMENT;

  if (fread_unlocked (buf, 1, strlen (blah) - 2, fp) != strlen (blah) - 2)
    {
      puts ("Could not read string from file");
      exit (1);
    }

  if (getc_unlocked (fp) != 'A')
    {
      puts ("getc_unlocked failed");
      exit (1);
    }

  f = fp;
  if (fgetc_unlocked (f++) != 'H'
      || f != fp + 1
      || fgetc_unlocked (--f) != EOF
      || f != fp)
    {
      puts ("fgetc_unlocked failed");
      exit (1);
    }

  if (ftello (fp) != (off_t) strlen (blah))
    {
      printf ("Failed to read %zd bytes from temporary file", strlen (blah));
      exit (1);
    }

  funlockfile (fp);
  fclose (fp);

  return 0;
}

static void
do_prepare (void)
{
  fd = create_temp_file ("tst-unlockedio.", NULL);
  if (fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }
}
