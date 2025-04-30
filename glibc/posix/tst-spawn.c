/* Tests for spawn.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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
#include <getopt.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#include <support/check.h>
#include <support/xunistd.h>
#include <support/temp_file.h>
#include <support/support.h>


/* Nonzero if the program gets called via `exec'.  */
static int restart;


#define CMDLINE_OPTIONS \
  { "restart", no_argument, &restart, 1 },

/* Name of the temporary files.  */
static char *name1;
static char *name2;
static char *name3;
static char *name5;

/* Descriptors for the temporary files.  */
static int temp_fd1 = -1;
static int temp_fd2 = -1;
static int temp_fd3 = -1;
static int temp_fd5 = -1;

/* The contents of our files.  */
static const char fd1string[] = "This file should get closed";
static const char fd2string[] = "This file should stay opened";
static const char fd3string[] = "This file will be opened";
static const char fd5string[] = "This file should stay opened (O_CLOEXEC)";


/* We have a preparation function.  */
static void
do_prepare (int argc, char *argv[])
{
  /* We must not open any files in the restart case.  */
  if (restart)
    return;

  TEST_VERIFY_EXIT ((temp_fd1 = create_temp_file ("spawn", &name1)) != -1);
  TEST_VERIFY_EXIT ((temp_fd2 = create_temp_file ("spawn", &name2)) != -1);
  TEST_VERIFY_EXIT ((temp_fd3 = create_temp_file ("spawn", &name3)) != -1);
  TEST_VERIFY_EXIT ((temp_fd5 = create_temp_file ("spawn", &name5)) != -1);

  int flags;
  TEST_VERIFY_EXIT ((flags = fcntl (temp_fd5, F_GETFD, &flags)) != -1);
  TEST_COMPARE (fcntl (temp_fd5, F_SETFD, flags | FD_CLOEXEC), 0);
}
#define PREPARE do_prepare


static int
handle_restart (const char *fd1s, const char *fd2s, const char *fd3s,
		const char *fd4s, const char *name, const char *fd5s)
{
  char buf[100];
  int fd1;
  int fd2;
  int fd3;
  int fd4;
  int fd5;

  /* First get the descriptors.  */
  fd1 = atol (fd1s);
  fd2 = atol (fd2s);
  fd3 = atol (fd3s);
  fd4 = atol (fd4s);
  fd5 = atol (fd5s);

  /* Sanity check.  */
  TEST_VERIFY_EXIT (fd1 != fd2);
  TEST_VERIFY_EXIT (fd1 != fd3);
  TEST_VERIFY_EXIT (fd1 != fd4);
  TEST_VERIFY_EXIT (fd2 != fd3);
  TEST_VERIFY_EXIT (fd2 != fd4);
  TEST_VERIFY_EXIT (fd3 != fd4);
  TEST_VERIFY_EXIT (fd4 != fd5);

  /* First the easy part: read from the file descriptor which is
     supposed to be open.  */
  TEST_COMPARE (xlseek (fd2, 0, SEEK_CUR), strlen (fd2string));
  /* The duped descriptor must have the same position.  */
  TEST_COMPARE (xlseek (fd4, 0, SEEK_CUR), strlen (fd2string));
  TEST_COMPARE (xlseek (fd2, 0, SEEK_SET), 0);
  TEST_COMPARE (xlseek (fd4, 0, SEEK_CUR), 0);
  TEST_COMPARE (read (fd2, buf, sizeof buf), strlen (fd2string));
  TEST_COMPARE_BLOB (fd2string, strlen (fd2string), buf, strlen (fd2string));

  /* Now read from the third file.  */
  TEST_COMPARE (read (fd3, buf, sizeof buf), strlen (fd3string));
  TEST_COMPARE_BLOB (fd3string, strlen (fd3string), buf, strlen (fd3string));
  /* Try to write to the file.  This should not be allowed.  */
  TEST_COMPARE (write (fd3, "boo!", 4), -1);
  TEST_COMPARE (errno, EBADF);

  /* Now try to read the first file.  First make sure it is not opened.  */
  TEST_COMPARE (lseek (fd1, 0, SEEK_CUR), (off_t) -1);
  TEST_COMPARE (errno, EBADF);

  /* Now open the file and read it.  */
  fd1 = xopen (name, O_RDONLY, 0600);

  TEST_COMPARE (read (fd1, buf, sizeof buf), strlen (fd1string));
  TEST_COMPARE_BLOB (fd1string, strlen (fd1string), buf, strlen (fd1string));

  TEST_COMPARE (xlseek (fd5, 0, SEEK_SET), 0);
  TEST_COMPARE (read (fd5, buf, sizeof buf), strlen (fd5string));
  TEST_COMPARE_BLOB (fd5string, strlen (fd5string), buf, strlen (fd5string));

  return 0;
}


static int
do_test (int argc, char *argv[])
{
  pid_t pid;
  int fd4;
  int status;
  posix_spawn_file_actions_t actions;
  char fd1name[18];
  char fd2name[18];
  char fd3name[18];
  char fd4name[18];
  char fd5name[18];
  char *name3_copy;
  char *spargv[13];
  int i;

  /* We must have
     - one or four parameters left if called initially
       + path for ld.so		optional
       + "--library-path"	optional
       + the library path	optional
       + the application name
     - six parameters left if called through re-execution
       + file descriptor number which is supposed to be closed
       + the open file descriptor
       + the newly opened file descriptor
       + the duped second descriptor
       + the name of the closed descriptor
       + the duped fourth file descriptor which O_CLOEXEC should be
	 remove by adddup2.
  */
  if (argc != (restart ? 7 : 2) && argc != (restart ? 7 : 5))
    FAIL_EXIT1 ("wrong number of arguments (%d)", argc);

  if (restart)
    return handle_restart (argv[1], argv[2], argv[3], argv[4], argv[5],
			   argv[6]);

  /* Prepare the test.  We are creating four files: two which file descriptor
     will be marked with FD_CLOEXEC, another which is not.  */

  /* Write something in the files.  */
  xwrite (temp_fd1, fd1string, strlen (fd1string));
  xwrite (temp_fd2, fd2string, strlen (fd2string));
  xwrite (temp_fd3, fd3string, strlen (fd3string));
  xwrite (temp_fd5, fd5string, strlen (fd5string));

  /* Close the third file.  It'll be opened by `spawn'.  */
  xclose (temp_fd3);

  /* Tell `spawn' what to do.  */
  TEST_COMPARE (posix_spawn_file_actions_init (&actions), 0);
  /* Close `temp_fd1'.  */
  TEST_COMPARE (posix_spawn_file_actions_addclose (&actions, temp_fd1), 0);
  /* We want to open the third file.  */
  name3_copy = xstrdup (name3);
  TEST_COMPARE (posix_spawn_file_actions_addopen (&actions, temp_fd3,
						  name3_copy,
						  O_RDONLY, 0666),
		0);
  /* Overwrite the name to check that a copy has been made.  */
  memset (name3_copy, 'X', strlen (name3_copy));

  /* We dup the second descriptor.  */
  fd4 = MAX (2, MAX (temp_fd1, MAX (temp_fd2, MAX (temp_fd3, temp_fd5)))) + 1;
  TEST_COMPARE (posix_spawn_file_actions_adddup2 (&actions, temp_fd2, fd4),
	        0);

  /* We clear the O_CLOEXEC on fourth descriptor, so it should be
     stay open on child.  */
  TEST_COMPARE (posix_spawn_file_actions_adddup2 (&actions, temp_fd5,
						  temp_fd5),
		0);

  /* Now spawn the process.  */
  snprintf (fd1name, sizeof fd1name, "%d", temp_fd1);
  snprintf (fd2name, sizeof fd2name, "%d", temp_fd2);
  snprintf (fd3name, sizeof fd3name, "%d", temp_fd3);
  snprintf (fd4name, sizeof fd4name, "%d", fd4);
  snprintf (fd5name, sizeof fd5name, "%d", temp_fd5);

  for (i = 0; i < (argc == (restart ? 7 : 5) ? 4 : 1); i++)
    spargv[i] = argv[i + 1];
  spargv[i++] = (char *) "--direct";
  spargv[i++] = (char *) "--restart";
  spargv[i++] = fd1name;
  spargv[i++] = fd2name;
  spargv[i++] = fd3name;
  spargv[i++] = fd4name;
  spargv[i++] = name1;
  spargv[i++] = fd5name;
  spargv[i] = NULL;

  TEST_COMPARE (posix_spawn (&pid, argv[1], &actions, NULL, spargv, environ),
		0);

  /* Wait for the children.  */
  TEST_COMPARE (xwaitpid (pid, &status, 0), pid);
  TEST_VERIFY (WIFEXITED (status));
  TEST_VERIFY (!WIFSIGNALED (status));
  TEST_COMPARE (WEXITSTATUS (status), 0);

  /* Same test but with a NULL pid argument.  */
  TEST_COMPARE (posix_spawn (NULL, argv[1], &actions, NULL, spargv, environ),
		0);

  /* Cleanup.  */
  TEST_COMPARE (posix_spawn_file_actions_destroy (&actions), 0);
  free (name3_copy);

  /* Wait for the children.  */
  xwaitpid (-1, &status, 0);
  TEST_VERIFY (WIFEXITED (status));
  TEST_VERIFY (!WIFSIGNALED (status));
  TEST_COMPARE (WEXITSTATUS (status), 0);

  return 0;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
