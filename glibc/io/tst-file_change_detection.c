/* Test for <file_change_detection.c>.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <file_change_detection.h>

#include <array_length.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xstdio.h>
#include <support/xunistd.h>
#include <unistd.h>

static void
all_same (struct file_change_detection *array, size_t length)
{
  for (size_t i = 0; i < length; ++i)
    for (size_t j = 0; j < length; ++j)
      {
        if (test_verbose > 0)
          printf ("info: comparing %zu and %zu\n", i, j);
        TEST_VERIFY (__file_is_unchanged (array + i, array + j));
      }
}

static void
all_different (struct file_change_detection *array, size_t length)
{
  for (size_t i = 0; i < length; ++i)
    for (size_t j = 0; j < length; ++j)
      {
        if (i == j)
          continue;
        if (test_verbose > 0)
          printf ("info: comparing %zu and %zu\n", i, j);
        TEST_VERIFY (!__file_is_unchanged (array + i, array + j));
      }
}

static int
do_test (void)
{
  /* Use a temporary directory with various paths.  */
  char *tempdir = support_create_temp_directory ("tst-file_change_detection-");

  char *path_dangling = xasprintf ("%s/dangling", tempdir);
  char *path_does_not_exist = xasprintf ("%s/does-not-exist", tempdir);
  char *path_empty1 = xasprintf ("%s/empty1", tempdir);
  char *path_empty2 = xasprintf ("%s/empty2", tempdir);
  char *path_fifo = xasprintf ("%s/fifo", tempdir);
  char *path_file1 = xasprintf ("%s/file1", tempdir);
  char *path_file2 = xasprintf ("%s/file2", tempdir);
  char *path_loop = xasprintf ("%s/loop", tempdir);
  char *path_to_empty1 = xasprintf ("%s/to-empty1", tempdir);
  char *path_to_file1 = xasprintf ("%s/to-file1", tempdir);

  add_temp_file (path_dangling);
  add_temp_file (path_empty1);
  add_temp_file (path_empty2);
  add_temp_file (path_fifo);
  add_temp_file (path_file1);
  add_temp_file (path_file2);
  add_temp_file (path_loop);
  add_temp_file (path_to_empty1);
  add_temp_file (path_to_file1);

  xsymlink ("target-does-not-exist", path_dangling);
  support_write_file_string (path_empty1, "");
  support_write_file_string (path_empty2, "");
  TEST_COMPARE (mknod (path_fifo, 0777 | S_IFIFO, 0), 0);
  support_write_file_string (path_file1, "line\n");
  support_write_file_string (path_file2, "line\n");
  xsymlink ("loop", path_loop);
  xsymlink ("empty1", path_to_empty1);
  xsymlink ("file1", path_to_file1);

  FILE *fp_file1 = xfopen (path_file1, "r");
  FILE *fp_file2 = xfopen (path_file2, "r");
  FILE *fp_empty1 = xfopen (path_empty1, "r");
  FILE *fp_empty2 = xfopen (path_empty2, "r");

  /* Test for the same (empty) files.  */
  {
    struct file_change_detection fcd[10];
    int i = 0;
    /* Two empty files always have the same contents.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_empty1));
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_empty2));
    /* So does a missing file (which is treated as empty).  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++],
                                                   path_does_not_exist));
    /* And a symbolic link loop.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_loop));
    /* And a dangling symbolic link.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_dangling));
    /* And a directory.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], tempdir));
    /* And a symbolic link to an empty file.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_to_empty1));
    /* Likewise for access the file via a FILE *.  */
    TEST_VERIFY (__file_change_detection_for_fp (&fcd[i++], fp_empty1));
    TEST_VERIFY (__file_change_detection_for_fp (&fcd[i++], fp_empty2));
    /* And a NULL FILE * (missing file).  */
    TEST_VERIFY (__file_change_detection_for_fp (&fcd[i++], NULL));
    TEST_COMPARE (i, array_length (fcd));

    all_same (fcd, array_length (fcd));
  }

  /* Symbolic links are resolved.  */
  {
    struct file_change_detection fcd[3];
    int i = 0;
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_file1));
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_to_file1));
    TEST_VERIFY (__file_change_detection_for_fp (&fcd[i++], fp_file1));
    TEST_COMPARE (i, array_length (fcd));
    all_same (fcd, array_length (fcd));
  }

  /* Test for different files.  */
  {
    struct file_change_detection fcd[5];
    int i = 0;
    /* The other files are not empty.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_empty1));
    /* These two files have the same contents, but have different file
       identity.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_file1));
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_file2));
    /* FIFOs are always different, even with themselves.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_fifo));
    TEST_VERIFY (__file_change_detection_for_path (&fcd[i++], path_fifo));
    TEST_COMPARE (i, array_length (fcd));
    all_different (fcd, array_length (fcd));

    /* Replacing the file with its symbolic link does not make a
       difference.  */
    TEST_VERIFY (__file_change_detection_for_path (&fcd[1], path_to_file1));
    all_different (fcd, array_length (fcd));
  }

  /* Wait for a file change.  Depending on file system time stamp
     resolution, this subtest blocks for a while.  */
  for (int use_stdio = 0; use_stdio < 2; ++use_stdio)
    {
      struct file_change_detection initial;
      TEST_VERIFY (__file_change_detection_for_path (&initial, path_file1));
      while (true)
        {
          support_write_file_string (path_file1, "line\n");
          struct file_change_detection current;
          if (use_stdio)
            TEST_VERIFY (__file_change_detection_for_fp (&current, fp_file1));
          else
            TEST_VERIFY (__file_change_detection_for_path
                         (&current, path_file1));
          if (!__file_is_unchanged (&initial, &current))
            break;
          /* Wait for a bit to reduce system load.  */
          usleep (100 * 1000);
        }
    }

  fclose (fp_empty1);
  fclose (fp_empty2);
  fclose (fp_file1);
  fclose (fp_file2);

  free (path_dangling);
  free (path_does_not_exist);
  free (path_empty1);
  free (path_empty2);
  free (path_fifo);
  free (path_file1);
  free (path_file2);
  free (path_loop);
  free (path_to_empty1);
  free (path_to_file1);

  free (tempdir);

  return 0;
}

#include <support/test-driver.c>
