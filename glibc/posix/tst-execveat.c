/* Test execveat at the various corner cases.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xdlfcn.h>
#include <support/xstdio.h>
#include <support/xunistd.h>
#include <wait.h>
#include <support/test-driver.h>

int
call_execveat (int fd, const char *pathname, int flags, int expected_fail,
               int num)
{
  char *envp[] = { (char *) "FOO=3", NULL };
  char *argv[] = { (char *) "sh", (char *) "-c", (char *) "exit $FOO", NULL };
  pid_t pid;
  int status;

  if (test_verbose > 0)
    printf ("call line number: %d\n", num);

  pid = xfork ();
  if (pid == 0)
    {
      TEST_COMPARE (execveat (fd, pathname, argv, envp, flags), -1);
      if (errno == ENOSYS)
	exit (EXIT_UNSUPPORTED);
      else if (errno == expected_fail)
        {
          if (test_verbose > 0)
            printf ("expected fail: errno %d\n", errno);
          _exit (0);
        }
      else
        FAIL_EXIT1 ("execveat failed: %m (%d)", errno);
    }
  xwaitpid (pid, &status, 0);

  if (!WIFEXITED (status))
    FAIL_RET ("child hasn't exited normally");

  if (WIFEXITED (status))
    {
      if (WEXITSTATUS (status) == EXIT_UNSUPPORTED)
        FAIL_UNSUPPORTED ("execveat is unimplemented");
      else if (expected_fail != 0)
        TEST_COMPARE (WEXITSTATUS (status), 0);
      else
        TEST_COMPARE (WEXITSTATUS (status), 3);
    }
  return 0;
}

static int
do_test (void)
{
  DIR *dirp;
  int fd;
#ifdef O_PATH
  int fd_out;
  char *tmp_dir, *symlink_name, *tmp_sh;
  struct stat64 st;
#endif

  dirp = opendir ("/bin");
  if (dirp == NULL)
    FAIL_EXIT1 ("failed to open /bin");
  fd = dirfd (dirp);

  /* Call execveat for various fd/pathname combinations.  */

  /* Check the pathname relative to a valid dirfd.  */
  call_execveat (fd, "sh", 0, 0, __LINE__);
  xchdir ("/bin");
  /* Use the special value AT_FDCWD as dirfd. Quoting open(2):
     If pathname is relative and dirfd is the special value AT_FDCWD, then
     pathname is interpreted relative to the current working directory of
     the calling process.  */
  call_execveat (AT_FDCWD, "sh", 0, 0, __LINE__);
  xclose (fd);
#ifdef O_PATH
  /* Check the pathname relative to a valid dirfd with O_PATH.  */
  fd = xopen ("/bin", O_PATH | O_DIRECTORY, O_RDONLY);
  call_execveat (fd, "sh", 0, 0, __LINE__);
  xclose (fd);

  /* Check absolute pathname, dirfd should be ignored.  */
  call_execveat (AT_FDCWD, "/bin/sh", 0, 0, __LINE__);
  fd = xopen ("/usr", O_PATH | O_DIRECTORY, 0);
  /* Same check for absolute pathname, but with input file descriptor
     openend with different flags.  The dirfd should be ignored.  */
  call_execveat (fd, "/bin/sh", 0, 0, __LINE__);
  xclose (fd);
#endif

  fd = xopen ("/usr", O_RDONLY, 0);
  /* Same check for absolute pathname, but with input file descriptor
     openend with different flags.  The dirfd should be ignored.  */
  call_execveat (fd, "/bin/sh", 0, 0, __LINE__);
  xclose (fd);

  fd = xopen ("/bin/sh", O_RDONLY, 0);
  /* Check relative pathname, where dirfd does not point to a directory.  */
  call_execveat (fd, "sh", 0, ENOTDIR, __LINE__);
  /* Check absolute pathname, but dirfd is a regular file.  The dirfd
     should be ignored.  */
  call_execveat (fd, "/bin/sh", 0, 0, __LINE__);
  xclose (fd);

#ifdef O_PATH
  /* Quoting open(2): O_PATH
     Obtain a file descriptor that can be used for two purposes: to
     indicate a location in the filesystem tree and to perform
     operations that act purely at the file descriptor level.  */
  fd = xopen ("/bin/sh", O_PATH, 0);
  /* Check the empty pathname.  Dirfd is a regular file with O_PATH.  */
  call_execveat (fd, "", 0, ENOENT, __LINE__);
  /* Same check for an empty pathname, but with AT_EMPTY_PATH flag.
     Quoting open(2):
     If oldpath is an empty string, create a link to the file referenced
     by olddirfd (which may have been obtained using the open(2) O_PATH flag. */
  call_execveat (fd, "", AT_EMPTY_PATH, 0, __LINE__);
  call_execveat (fd, "", AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW, 0, __LINE__);
  xclose (fd);

  /* Create a temporary directory "tmp_dir" and create a symbolik link tmp_sh
     pointing to /bin/sh inside the tmp_dir. Open dirfd as a symbolic link.  */
  tmp_dir = support_create_temp_directory ("tst-execveat_dir");
  symlink_name = xasprintf ("%s/symlink", tmp_dir);
  xsymlink ("tmp_sh", symlink_name);
  add_temp_file (symlink_name);
  tmp_sh = xasprintf ("%s/tmp_sh", tmp_dir);
  add_temp_file (tmp_sh);
  fd_out = xopen (symlink_name, O_CREAT | O_WRONLY, 0);
  xstat ("/bin/sh", &st);
  fd = xopen ("/bin/sh", O_RDONLY, 0);
  xcopy_file_range (fd, 0, fd_out, 0, st.st_size, 0);
  xfchmod (fd_out, 0700);
  xclose (fd);
  xclose (fd_out);
  fd_out = xopen (symlink_name, O_PATH, 0);

 /* Check the empty pathname. Dirfd is a symbolic link.  */
  call_execveat (fd_out, "", 0, ENOENT, __LINE__);
  call_execveat (fd_out, "", AT_EMPTY_PATH, 0, __LINE__);
  call_execveat (fd_out, "", AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW, 0,
                 __LINE__);
  xclose (fd_out);
  free (symlink_name);
  free (tmp_sh);
  free (tmp_dir);
#endif

  /* Call execveat with closed fd, we expect this to fail with EBADF.  */
  call_execveat (fd, "sh", 0, EBADF, __LINE__);
  /* Call execveat with closed fd, we expect this to pass because the pathname is
     absolute.  */
  call_execveat (fd, "/bin/sh", 0, 0, __LINE__);

  return 0;
}

#include <support/test-driver.c>
