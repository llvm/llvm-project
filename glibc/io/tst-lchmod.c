/* Tests for lchmod and fchmodat with AT_SYMLINK_NOFOLLOW.
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

#include <array_length.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/descriptors.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xunistd.h>
#include <unistd.h>

#if __has_include (<sys/mount.h>)
# include <sys/mount.h>
#endif

/* Array of file descriptors.  */
#define DYNARRAY_STRUCT fd_list
#define DYNARRAY_ELEMENT int
#define DYNARRAY_INITIAL_SIZE 0
#define DYNARRAY_PREFIX fd_list_
#include <malloc/dynarray-skeleton.c>

static int
fchmodat_with_lchmod (int fd, const char *path, mode_t mode, int flags)
{
  TEST_COMPARE (fd, AT_FDCWD);
  if (flags == 0)
    return chmod (path, mode);
  else
    {
      TEST_COMPARE (flags, AT_SYMLINK_NOFOLLOW);
      return lchmod (path, mode);
    }
}

/* Chose the appropriate path to pass as the path argument to the *at
   functions.  */
static const char *
select_path (bool do_relative_path, const char *full_path, const char *relative_path)
{
  if (do_relative_path)
    return relative_path;
  else
    return full_path;
}

static void
test_1 (bool do_relative_path, int (*chmod_func) (int fd, const char *, mode_t, int))
{
  char *tempdir = support_create_temp_directory ("tst-lchmod-");

  char *path_dangling = xasprintf ("%s/dangling", tempdir);
  char *path_file = xasprintf ("%s/file", tempdir);
  char *path_loop = xasprintf ("%s/loop", tempdir);
  char *path_missing = xasprintf ("%s/missing", tempdir);
  char *path_to_file = xasprintf ("%s/to-file", tempdir);

  int fd;
  if (do_relative_path)
    fd = xopen (tempdir, O_DIRECTORY | O_RDONLY, 0);
  else
    fd = AT_FDCWD;

  add_temp_file (path_dangling);
  add_temp_file (path_loop);
  add_temp_file (path_file);
  add_temp_file (path_to_file);

  support_write_file_string (path_file, "");
  xsymlink ("file", path_to_file);
  xsymlink ("loop", path_loop);
  xsymlink ("target-does-not-exist", path_dangling);

  /* Check that the modes do not collide with what we will use in the
     test.  */
  struct stat64 st;
  xstat (path_file, &st);
  TEST_VERIFY ((st.st_mode & 0777) != 1);
  xlstat (path_to_file, &st);
  TEST_VERIFY ((st.st_mode & 0777) != 2);
  mode_t original_symlink_mode = st.st_mode;

  /* We should be able to change the mode of a file, including through
     the symbolic link to-file.  */
  const char *arg = select_path (do_relative_path, path_file, "file");
  TEST_COMPARE (chmod_func (fd, arg, 1, 0), 0);
  xstat (path_file, &st);
  TEST_COMPARE (st.st_mode & 0777, 1);
  arg = select_path (do_relative_path, path_to_file, "to-file");
  TEST_COMPARE (chmod_func (fd, arg, 2, 0), 0);
  xstat (path_file, &st);
  TEST_COMPARE (st.st_mode & 0777, 2);
  xlstat (path_to_file, &st);
  TEST_COMPARE (original_symlink_mode, st.st_mode);
  arg = select_path (do_relative_path, path_file, "file");
  TEST_COMPARE (chmod_func (fd, arg, 1, 0), 0);
  xstat (path_file, &st);
  TEST_COMPARE (st.st_mode & 0777, 1);
  xlstat (path_to_file, &st);
  TEST_COMPARE (original_symlink_mode, st.st_mode);

  /* Changing the mode of a symbolic link should fail.  */
  arg = select_path (do_relative_path, path_to_file, "to-file");
  int ret = chmod_func (fd, arg, 2, AT_SYMLINK_NOFOLLOW);
  TEST_COMPARE (ret, -1);
  TEST_COMPARE (errno, EOPNOTSUPP);

  /* The modes should remain unchanged.  */
  xstat (path_file, &st);
  TEST_COMPARE (st.st_mode & 0777, 1);
  xlstat (path_to_file, &st);
  TEST_COMPARE (original_symlink_mode, st.st_mode);

  /* Likewise, changing dangling and looping symbolic links must
     fail.  */
  const char *paths[] = { path_dangling, path_loop };
  for (size_t i = 0; i < array_length (paths); ++i)
    {
      const char *path = paths[i];
      const char *filename = strrchr (path, '/');
      TEST_VERIFY_EXIT (filename != NULL);
      ++filename;
      mode_t new_mode = 010 + i;

      xlstat (path, &st);
      TEST_VERIFY ((st.st_mode & 0777) != new_mode);
      original_symlink_mode = st.st_mode;
      arg = select_path (do_relative_path, path, filename);
      ret = chmod_func (fd, arg, new_mode, AT_SYMLINK_NOFOLLOW);
      TEST_COMPARE (ret, -1);
      TEST_COMPARE (errno, EOPNOTSUPP);
      xlstat (path, &st);
      TEST_COMPARE (st.st_mode, original_symlink_mode);
    }

   /* A missing file should always result in ENOENT.  The presence of
      /proc does not matter.  */
   arg = select_path (do_relative_path, path_missing, "missing");
   TEST_COMPARE (chmod_func (fd, arg, 020, 0), -1);
   TEST_COMPARE (errno, ENOENT);
   TEST_COMPARE (chmod_func (fd, arg, 020, AT_SYMLINK_NOFOLLOW), -1);
   TEST_COMPARE (errno, ENOENT);

   /* Test without available file descriptors.  */
   {
     struct fd_list fd_list;
     fd_list_init (&fd_list);
     while (true)
       {
         int ret = dup (STDOUT_FILENO);
         if (ret == -1)
           {
             if (errno == ENFILE || errno == EMFILE)
               break;
             FAIL_EXIT1 ("dup: %m");
           }
         fd_list_add (&fd_list, ret);
         TEST_VERIFY_EXIT (!fd_list_has_failed (&fd_list));
       }
     /* Without AT_SYMLINK_NOFOLLOW, changing the permissions should
        work as before.  */
     arg = select_path (do_relative_path, path_file, "file");
     TEST_COMPARE (chmod_func (fd, arg, 3, 0), 0);
     xstat (path_file, &st);
     TEST_COMPARE (st.st_mode & 0777, 3);
     /* But with AT_SYMLINK_NOFOLLOW, even if we originally had
        support, we may have lost it.  */
     ret = chmod_func (fd, arg, 2, AT_SYMLINK_NOFOLLOW);
     if (ret == 0)
       {
         xstat (path_file, &st);
         TEST_COMPARE (st.st_mode & 0777, 2);
       }
     else
       {
         TEST_COMPARE (ret, -1);
         /* The error code from the openat fallback leaks out.  */
         if (errno != ENFILE && errno != EMFILE)
           TEST_COMPARE (errno, EOPNOTSUPP);
       }
     xstat (path_file, &st);
     TEST_COMPARE (st.st_mode & 0777, 3);

     /* Close the descriptors.  */
     for (int *pfd = fd_list_begin (&fd_list); pfd < fd_list_end (&fd_list);
          ++pfd)
       xclose (*pfd);
     fd_list_free (&fd_list);
   }

   if (do_relative_path)
    xclose (fd);

   free (path_dangling);
   free (path_file);
   free (path_loop);
   free (path_missing);
   free (path_to_file);

   free (tempdir);
}

static void
test_3 (void)
{
  puts ("info: testing lchmod");
  test_1 (false, fchmodat_with_lchmod);
  puts ("info: testing fchmodat with AT_FDCWD");
  test_1 (false, fchmodat);
  puts ("info: testing fchmodat with relative path");
  test_1 (true, fchmodat);
}

static int
do_test (void)
{
  struct support_descriptors *descriptors = support_descriptors_list ();

  /* Run the three tests in the default environment.  */
  test_3 ();

  /* Try to set up a /proc-less environment and re-test.  */
#if __has_include (<sys/mount.h>)
  if (!support_become_root ())
    puts ("warning: could not obtain root-like privileges");
  if (!support_enter_mount_namespace ())
    puts ("warning: could enter a mount namespace");
  else
    {
      /* Attempt to mount an empty directory over /proc.  */
      char *tempdir = support_create_temp_directory ("tst-lchmod-");
      bool proc_emptied
        = mount (tempdir, "/proc", "none", MS_BIND, NULL) == 0;
      if (!proc_emptied)
        printf ("warning: bind-mounting /proc failed: %m");
      free (tempdir);

      puts ("info: re-running tests (after trying to empty /proc)");
      test_3 ();

      if (proc_emptied)
        /* Reveal the original /proc, which is needed by the
           descriptors check below.  */
        TEST_COMPARE (umount ("/proc"), 0);
    }
#endif /* <sys/mount.h>.  */

  support_descriptors_check (descriptors);
  support_descriptors_free (descriptors);

  return 0;
}

#include <support/test-driver.c>
