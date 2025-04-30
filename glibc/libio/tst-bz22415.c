/* Check static buffer handling with setvbuf (BZ #22415)

   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <wchar.h>
#include <mcheck.h>

#include <support/check.h>
#include <support/temp_file.h>

static int
do_test (void)
{
  mtrace ();

  char *temp_file;
  TEST_VERIFY_EXIT (create_temp_file ("tst-bz22145.", &temp_file));

  char buf[BUFSIZ];

  {
    /* Check if backup buffer is correctly freed and changing back
       to normal buffer does not trigger an invalid free in case of
       static buffer set by setvbuf.  */

    FILE *f = fopen (temp_file, "w+b");
    TEST_VERIFY_EXIT (f != NULL);

    TEST_VERIFY_EXIT (setvbuf (f, buf, _IOFBF, BUFSIZ) == 0);
    TEST_VERIFY_EXIT (ungetc ('x', f) == 'x');
    TEST_VERIFY_EXIT (fseek (f, 0L, SEEK_SET) == 0);
    TEST_VERIFY_EXIT (fputc ('y', f) ==  'y');

    TEST_VERIFY_EXIT (fclose (f) == 0);
  }

  {
    /* Check if backup buffer is correctly freed and changing back
       to normal buffer does not trigger an invalid free in case of
       static buffer set by setvbuf.  */

    FILE *f = fopen (temp_file, "w+b");
    TEST_VERIFY_EXIT (f != NULL);

    TEST_VERIFY_EXIT (setvbuf (f, buf, _IOFBF, BUFSIZ) == 0);
    TEST_VERIFY_EXIT (ungetc ('x', f) == 'x');
    TEST_VERIFY_EXIT (fputc ('y', f) ==  'y');

    TEST_VERIFY_EXIT (fclose (f) == 0);
  }

  {
    FILE *f = fopen (temp_file, "w+b");
    TEST_VERIFY_EXIT (f != NULL);

    TEST_VERIFY_EXIT (setvbuf (f, buf, _IOFBF, BUFSIZ) == 0);
    TEST_VERIFY_EXIT (ungetwc (L'x', f) == L'x');
    TEST_VERIFY_EXIT (fseek (f, 0L, SEEK_SET) == 0);
    TEST_VERIFY_EXIT (fputwc (L'y', f) ==  L'y');

    TEST_VERIFY_EXIT (fclose (f) == 0);
  }

  {
    FILE *f = fopen (temp_file, "w+b");
    TEST_VERIFY_EXIT (f != NULL);

    TEST_VERIFY_EXIT (setvbuf (f, buf, _IOFBF, BUFSIZ) == 0);
    TEST_VERIFY_EXIT (ungetwc (L'x', f) == L'x');
    TEST_VERIFY_EXIT (fputwc (L'y', f) ==  L'y');

    TEST_VERIFY_EXIT (fclose (f) == 0);
  }

  free (temp_file);

  return 0;
}

#include <support/test-driver.c>
