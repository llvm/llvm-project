/* Test open and openat with O_TMPFILE.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* This test verifies that open and openat work as expected, i.e. they
   create a deleted file with the requested file mode.  */

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <support/support.h>

#ifdef O_TMPFILE
typedef int (*wrapper_func) (const char *, int, mode_t);

/* Error-checking wrapper for the open function, compatible with the
   wrapper_func type.  */
static int
wrap_open (const char *path, int flags, mode_t mode)
{
  int ret = open (path, flags, mode);
  if (ret < 0)
    {
      printf ("error: open (\"%s\", 0x%x, 0%03o): %m\n", path, flags, mode);
      exit (1);
    }
  return ret;
}

/* Error-checking wrapper for the openat function, compatible with the
   wrapper_func type.  */
static int
wrap_openat (const char *path, int flags, mode_t mode)
{
  int ret = openat (AT_FDCWD, path, flags, mode);
  if (ret < 0)
    {
      printf ("error: openat (\"%s\", 0x%x, 0%03o): %m\n", path, flags, mode);
      exit (1);
    }
  return ret;
}

/* Error-checking wrapper for the open64 function, compatible with the
   wrapper_func type.  */
static int
wrap_open64 (const char *path, int flags, mode_t mode)
{
  int ret = open64 (path, flags, mode);
  if (ret < 0)
    {
      printf ("error: open64 (\"%s\", 0x%x, 0%03o): %m\n", path, flags, mode);
      exit (1);
    }
  return ret;
}

/* Error-checking wrapper for the openat64 function, compatible with the
   wrapper_func type.  */
static int
wrap_openat64 (const char *path, int flags, mode_t mode)
{
  int ret = openat64 (AT_FDCWD, path, flags, mode);
  if (ret < 0)
    {
      printf ("error: openat64 (\"%s\", 0x%x, 0%03o): %m\n", path, flags, mode);
      exit (1);
    }
  return ret;
}

/* Return true if FD is flagged as deleted in /proc/self/fd, false if
   not.  */
static bool
is_file_deteted (int fd)
{
  char *proc_fd_path = xasprintf ("/proc/self/fd/%d", fd);
  char file_path[4096];
  ssize_t file_path_length
    = readlink (proc_fd_path, file_path, sizeof (file_path));
  if (file_path_length < 0)
    {
      printf ("error: readlink (\"%s\"): %m", proc_fd_path);
      free (proc_fd_path);
      exit (1);
    }
  free (proc_fd_path);
  if (file_path_length == sizeof (file_path))
    {
      printf ("error: path in /proc resolves to overlong file name: %.*s\n",
              (int) file_path_length, file_path);
      exit (1);
    }
  const char *deleted = " (deleted)";
  if (file_path_length < strlen (deleted))
    {
      printf ("error: path in /proc is too short: %.*s\n",
              (int) file_path_length, file_path);
      exit (1);
    }
  return memcmp (file_path + file_path_length - strlen (deleted),
              deleted, strlen (deleted)) == 0;
}

/* Obtain a file name which is difficult to guess.  */
static char *
get_random_name (void)
{
  unsigned long long bytes[2];
  int random_device = open ("/dev/urandom", O_RDONLY);
  if (random_device < 0)
    {
      printf ("error: open (\"/dev/urandom\"): %m\n");
      exit (1);
    }
  ssize_t ret = read (random_device, bytes, sizeof (bytes));
  if (ret < 0)
    {
      printf ("error: read (\"/dev/urandom\"): %m\n");
      exit (1);
    }
  if (ret != sizeof (bytes))
    {
      printf ("error: short read from /dev/urandom: %zd\n", ret);
      exit (1);
    }
  close (random_device);
  return xasprintf ("tst-open-tmpfile-%08llx%08llx.tmp", bytes[0], bytes[1]);
}

/* Check open/openat (as specified by OP and WRAPPER) with a specific
   PATH/FLAGS/MODE combination.  */
static void
check_wrapper_flags_mode (const char *op, wrapper_func wrapper,
                          const char *path, int flags, mode_t mode)
{
  int fd = wrapper (path, flags | O_TMPFILE, mode);
  struct stat64 st;
  if (fstat64 (fd, &st) != 0)
    {
      printf ("error: fstat64: %m\n");
      exit (1);
    }

  /* Verify that the mode was correctly processed.  */
  int actual_mode = st.st_mode & 0777;
  if (actual_mode != mode)
    {
      printf ("error: unexpected mode; expected 0%03o, actual 0%03o\n",
              mode, actual_mode);
      exit (1);
    }

  /* Check that the file is marked as deleted in /proc.  */
  if (!is_file_deteted (fd))
    {
      printf ("error: path in /proc is not marked as deleted\n");
      exit (1);
    }

  /* Check that the file can be turned into a regular file with
     linkat.  Open a file descriptor for the directory at PATH.  Use
     AT_FDCWD if PATH is ".", to exercise that functionality as
     well.  */
  int path_fd;
  if (strcmp (path, ".") == 0)
    path_fd = AT_FDCWD;
  else
    {
      path_fd = open (path, O_RDONLY | O_DIRECTORY);
      if (path_fd < 0)
        {
          printf ("error: open (\"%s\"): %m\n", path);
          exit (1);
        }
    }

  /* Use a hard-to-guess name for the new directory entry.  */
  char *new_name = get_random_name ();

  /* linkat does not require privileges if the path in /proc/self/fd
     is used.  */
  char *proc_fd_path = xasprintf ("/proc/self/fd/%d", fd);
  if (linkat (AT_FDCWD, proc_fd_path, path_fd, new_name,
              AT_SYMLINK_FOLLOW) == 0)
    {
      if (unlinkat (path_fd, new_name, 0) != 0 && errno != ENOENT)
        {
          printf ("error: unlinkat (\"%s/%s\"): %m\n", path, new_name);
          exit (1);
        }
    }
  else
    {
      /* linkat failed.  This is expected if O_EXCL was specified.  */
      if ((flags & O_EXCL) == 0)
        {
          printf ("error: linkat failed after %s (\"%s\", 0x%x, 0%03o): %m\n",
                  op, path, flags, mode);
          exit (1);
        }
    }

  free (proc_fd_path);
  free (new_name);
  if (path_fd != AT_FDCWD)
    close (path_fd);
  close (fd);
}

/* Check OP/WRAPPER with various flags at a specific PATH and
   MODE.  */
static void
check_wrapper_mode (const char *op, wrapper_func wrapper,
                    const char *path, mode_t mode)
{
  check_wrapper_flags_mode (op, wrapper, path, O_WRONLY, mode);
  check_wrapper_flags_mode (op, wrapper, path, O_WRONLY | O_EXCL, mode);
  check_wrapper_flags_mode (op, wrapper, path, O_RDWR, mode);
  check_wrapper_flags_mode (op, wrapper, path, O_RDWR | O_EXCL, mode);
}

/* Check open/openat with varying permissions.  */
static void
check_wrapper (const char *op, wrapper_func wrapper,
                    const char *path)
{
  printf ("info: testing %s at: %s\n", op, path);
  check_wrapper_mode (op, wrapper, path, 0);
  check_wrapper_mode (op, wrapper, path, 0640);
  check_wrapper_mode (op, wrapper, path, 0600);
  check_wrapper_mode (op, wrapper, path, 0755);
  check_wrapper_mode (op, wrapper, path, 0750);
}

/* Verify that the directory at PATH supports O_TMPFILE.  Exit with
   status 77 (unsupported) if the kernel does not support O_TMPFILE.
   Even with kernel support, not all file systems O_TMPFILE, so return
   true if the directory supports O_TMPFILE, false if not.  */
static bool
probe_path (const char *path)
{
  int fd = openat (AT_FDCWD, path, O_TMPFILE | O_RDWR, 0);
  if (fd < 0)
    {
      if (errno == EISDIR)
        /* The system does not support O_TMPFILE.  */
        {
          printf ("info: kernel does not support O_TMPFILE\n");
          exit (77);
        }
      if (errno == EOPNOTSUPP)
        {
          printf ("info: path does not support O_TMPFILE: %s\n", path);
          return false;
        }
      printf ("error: openat (\"%s\", O_TMPFILE | O_RDWR): %m\n", path);
      exit (1);
    }
  close (fd);
  return true;
}

static int
do_test (void)
{
  umask (0);
  const char *paths[] = { ".", "/dev/shm", "/tmp",
                          getenv ("TEST_TMPFILE_PATH"),
                          NULL };
  bool supported = false;
  for (int i = 0; paths[i] != NULL; ++i)
    if (probe_path (paths[i]))
      {
        supported = true;
        check_wrapper ("open", wrap_open, paths[i]);
        check_wrapper ("openat", wrap_openat, paths[i]);
        check_wrapper ("open64", wrap_open64, paths[i]);
        check_wrapper ("openat64", wrap_openat64, paths[i]);
      }

  if (!supported)
    return 77;

  return 0;
}

#else  /* !O_TMPFILE */

static int
do_test (void)
{
  return 77;
}

#endif  /* O_TMPFILE */

#include <support/test-driver.c>
