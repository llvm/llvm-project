/* Test for open_memstream BZ #21037.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include "tst-memstream.h"

static void
mcheck_abort (enum mcheck_status ev)
{
  printf ("mecheck failed with status %d\n", (int) ev);
  exit (1);
}

static int
do_test (void)
{
  mcheck_pedantic (mcheck_abort);

  /* Check if freopen proper fflush the stream.  */
  {
    CHAR_T old[] = W("old");
    CHAR_T *buf = old;
    size_t size;

    FILE *fp = OPEN_MEMSTREAM (&buf, &size);
    TEST_VERIFY_EXIT (fp != NULL);

    FPUTS (W("new"), fp);
    /* The stream buffer pointer should be updated with only a fflush or
       fclose.  */
    TEST_COMPARE (STRCMP (buf, old), 0);

    /* The old stream should be fflush the stream, even for an invalid
       streams.  */
    FILE *nfp = freopen ("invalid-file", "r", fp);
    TEST_VERIFY_EXIT (nfp == NULL);

    TEST_VERIFY (STRCMP (buf, W("new")) == 0);
    TEST_COMPARE_BLOB (buf, STRLEN (buf) * sizeof (CHAR_T),
		       W("new"), STRLEN (W("new")) * sizeof (CHAR_T));

    TEST_COMPARE (fclose (fp), 0);

    free (buf);
  }

  return 0;
}

#include <support/test-driver.c>
