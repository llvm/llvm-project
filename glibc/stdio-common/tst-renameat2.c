/* Linux implementation for renameat2 function.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <array_length.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xunistd.h>
#include <unistd.h>

/* Directory with the temporary files.  */
static char *directory;
static int directory_fd;

/* Paths within that directory.  */
static char *old_path;          /* File is called "old".  */
static char *new_path;          /* File is called "new".  */

/* Subdirectory within the directory above.  */
static char *subdirectory;
int subdirectory_fd;

/* And a pathname in that directory (called "file").  */
static char *subdir_path;

static void
prepare (int argc, char **argv)
{
  directory = support_create_temp_directory ("tst-renameat2-");
  directory_fd = xopen (directory, O_RDONLY | O_DIRECTORY, 0);
  old_path = xasprintf ("%s/old", directory);
  add_temp_file (old_path);
  new_path = xasprintf ("%s/new", directory);
  add_temp_file (new_path);
  subdirectory = xasprintf ("%s/subdir", directory);
  xmkdir (subdirectory, 0777);
  add_temp_file (subdirectory);
  subdirectory_fd = xopen (subdirectory, O_RDONLY | O_DIRECTORY, 0);
  subdir_path = xasprintf ("%s/file", subdirectory);
  add_temp_file (subdir_path);
}

/* Delete all files, preparing a clean slate for the next test.  */
static void
delete_all_files (void)
{
  char *files[] = { old_path, new_path, subdir_path };
  for (size_t i = 0; i < array_length (files); ++i)
    if (unlink (files[i]) != 0 && errno != ENOENT)
      FAIL_EXIT1 ("unlink (\"%s\"): %m", files[i]);
}

/* Return true if PATH exists in the file system.  */
static bool
file_exists (const char *path)
{
  return access (path, F_OK) == 0;
}

/* Check that PATH exists and has size EXPECTED_SIZE.  */
static void
check_size (const char *path, off64_t expected_size)
{
  struct stat64 st;
  xstat (path, &st);
  if (st.st_size != expected_size)
    FAIL_EXIT1 ("file \"%s\": expected size %lld, actual size %lld",
                path, (unsigned long long int) expected_size,
                (unsigned long long int) st.st_size);
}

/* Rename tests where the target does not exist.  */
static void
rename_without_existing_target (unsigned int flags)
{
  delete_all_files ();
  support_write_file_string (old_path, "");
  TEST_COMPARE (renameat2 (AT_FDCWD, old_path, AT_FDCWD, new_path, flags), 0);
  TEST_VERIFY (!file_exists (old_path));
  TEST_VERIFY (file_exists (new_path));

  delete_all_files ();
  support_write_file_string (old_path, "");
  TEST_COMPARE (renameat2 (directory_fd, "old", AT_FDCWD, new_path, flags), 0);
  TEST_VERIFY (!file_exists (old_path));
  TEST_VERIFY (file_exists (new_path));

  delete_all_files ();
  support_write_file_string (old_path, "");
  TEST_COMPARE (renameat2 (directory_fd, "old", subdirectory_fd, "file", 0),
                0);
  TEST_VERIFY (!file_exists (old_path));
  TEST_VERIFY (file_exists (subdir_path));
}

static int
do_test (void)
{
  /* Tests with zero flags argument.  These are expected to succeed
     because this renameat2 variant can be implemented with
     renameat.  */
  rename_without_existing_target (0);

  /* renameat2 without flags replaces an existing destination.  */
  delete_all_files ();
  support_write_file_string (old_path, "123");
  support_write_file_string (new_path, "1234");
  TEST_COMPARE (renameat2 (AT_FDCWD, old_path, AT_FDCWD, new_path, 0), 0);
  TEST_VERIFY (!file_exists (old_path));
  check_size (new_path, 3);

  /* Now we need to check for kernel support of renameat2 with
     flags.  */
  delete_all_files ();
  support_write_file_string (old_path, "");
  if (renameat2 (AT_FDCWD, old_path, AT_FDCWD, new_path, RENAME_NOREPLACE)
      != 0)
    {
      if (errno == EINVAL)
        puts ("warning: no support for renameat2 with flags");
      else
        FAIL_EXIT1 ("renameat2 probe failed: %m");
    }
  else
    {
      /* We have full renameat2 support.  */
      rename_without_existing_target (RENAME_NOREPLACE);

      /* Now test RENAME_NOREPLACE with an existing target.  */
      delete_all_files ();
      support_write_file_string (old_path, "123");
      support_write_file_string (new_path, "1234");
      TEST_COMPARE (renameat2 (AT_FDCWD, old_path, AT_FDCWD, new_path,
                               RENAME_NOREPLACE), -1);
      TEST_COMPARE (errno, EEXIST);
      check_size (old_path, 3);
      check_size (new_path, 4);

      delete_all_files ();
      support_write_file_string (old_path, "123");
      support_write_file_string (new_path, "1234");
      TEST_COMPARE (renameat2 (directory_fd, "old", AT_FDCWD, new_path,
                               RENAME_NOREPLACE), -1);
      TEST_COMPARE (errno, EEXIST);
      check_size (old_path, 3);
      check_size (new_path, 4);

      delete_all_files ();
      support_write_file_string (old_path, "123");
      support_write_file_string (subdir_path, "1234");
      TEST_COMPARE (renameat2 (directory_fd, "old", subdirectory_fd, "file",
                               RENAME_NOREPLACE), -1);
      TEST_COMPARE (errno, EEXIST);
      check_size (old_path, 3);
      check_size (subdir_path, 4);

      /* The flag combination of RENAME_NOREPLACE and RENAME_EXCHANGE
         is invalid.  */
      TEST_COMPARE (renameat2 (directory_fd, "ignored",
                               subdirectory_fd, "ignored",
                               RENAME_NOREPLACE | RENAME_EXCHANGE), -1);
      TEST_COMPARE (errno, EINVAL);
    }

  /* Create all the pathnames to avoid warnings from the test
     harness.  */
  support_write_file_string (old_path, "");
  support_write_file_string (new_path, "");
  support_write_file_string (subdir_path, "");

  free (directory);
  free (subdirectory);
  free (old_path);
  free (new_path);
  free (subdir_path);

  xclose (directory_fd);
  xclose (subdirectory_fd);

  return 0;
}

#define PREPARE prepare
#include <support/test-driver.c>
