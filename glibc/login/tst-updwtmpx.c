/* Basic test coverage for updwtmpx.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* This program runs a series of tests.  Each one calls updwtmpx
   twice, to write two records, optionally with misalignment in the
   file, and reads back the results.  */

#include <errno.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/descriptors.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xunistd.h>
#include <unistd.h>
#include <utmpx.h>

static int
do_test (void)
{
  /* Two entries filled with an arbitrary bit pattern.  */
  struct utmpx entries[2];
  unsigned char pad;
  {
    unsigned char *p = (unsigned char *) &entries[0];
    for (size_t i = 0; i < sizeof (entries); ++i)
      {
        p[i] = i;
      }
    /* Make sure that the first and second entry and the padding are
       different.  */
    p[sizeof (struct utmpx)] = p[0] + 1;
    pad = p[0] + 2;
  }

  char *path;
  int fd = create_temp_file ("tst-updwtmpx-", &path);

  /* Used to check that updwtmpx does not leave an open file
     descriptor around.  */
  struct support_descriptors *descriptors = support_descriptors_list ();

  /* updwtmpx is expected to remove misalignment.  Optionally insert
     one byte of misalignment at the start and in the middle (after
     the first entry).  */
  for (int misaligned_start = 0; misaligned_start < 2; ++misaligned_start)
    for (int misaligned_middle = 0; misaligned_middle < 2; ++misaligned_middle)
      {
        if (test_verbose > 0)
          printf ("info: misaligned_start=%d misaligned_middle=%d\n",
                  misaligned_start, misaligned_middle);

        xftruncate (fd, 0);
        TEST_COMPARE (pwrite64 (fd, &pad, misaligned_start, 0),
                      misaligned_start);

        /* Write first entry and check it.  */
        errno = 0;
        updwtmpx (path, &entries[0]);
        TEST_COMPARE (errno, 0);
        support_descriptors_check (descriptors);
        TEST_COMPARE (xlseek (fd, 0, SEEK_END), sizeof (struct utmpx));
        struct utmpx buffer;
        TEST_COMPARE (pread64 (fd, &buffer, sizeof (buffer), 0),
                      sizeof (buffer));
        TEST_COMPARE_BLOB (&entries[0], sizeof (entries[0]),
                           &buffer, sizeof (buffer));

        /* Middle mis-alignmet.  */
        TEST_COMPARE (pwrite64 (fd, &pad, misaligned_middle,
                                sizeof (struct utmpx)), misaligned_middle);

        /* Write second entry and check both entries.  */
        errno = 0;
        updwtmpx (path, &entries[1]);
        TEST_COMPARE (errno, 0);
        support_descriptors_check (descriptors);
        TEST_COMPARE (xlseek (fd, 0, SEEK_END), 2 * sizeof (struct utmpx));
        TEST_COMPARE (pread64 (fd, &buffer, sizeof (buffer), 0),
                      sizeof (buffer));
        TEST_COMPARE_BLOB (&entries[0], sizeof (entries[0]),
                           &buffer, sizeof (buffer));
        TEST_COMPARE (pread64 (fd, &buffer, sizeof (buffer), sizeof (buffer)),
                      sizeof (buffer));
        TEST_COMPARE_BLOB (&entries[1], sizeof (entries[1]),
                           &buffer, sizeof (buffer));
      }

  support_descriptors_free (descriptors);
  free (path);
  xclose (fd);

  return 0;
}

#include <support/test-driver.c>
