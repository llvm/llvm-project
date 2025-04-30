/* Tests for exec.
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

#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wait.h>


/* Nonzero if the program gets called via `exec'.  */
static int restart;


#define CMDLINE_OPTIONS \
  { "restart", no_argument, &restart, 1 },

/* Prototype for our test function.  */
extern void do_prepare (int argc, char *argv[]);
extern int do_test (int argc, char *argv[]);

/* We have a preparation function.  */
#define PREPARE do_prepare

#include "../test-skeleton.c"


/* Name of the temporary files.  */
static char *name1;
static char *name2;

/* File descriptors for these temporary files.  */
static int temp_fd1 = -1;
static int temp_fd2 = -1;

/* The contents of our files.  */
static const char fd1string[] = "This file should get closed";
static const char fd2string[] = "This file should stay opened";


/* We have a preparation function.  */
void
do_prepare (int argc, char *argv[])
{
  /* We must not open any files in the restart case.  */
  if (restart)
    return;

  temp_fd1 = create_temp_file ("exec", &name1);
  temp_fd2 = create_temp_file ("exec", &name2);
  if (temp_fd1 < 0 || temp_fd2 < 0)
    exit (1);
}


static int
handle_restart (const char *fd1s, const char *fd2s, const char *name)
{
  char buf[100];
  int fd1;
  int fd2;

  /* First get the descriptors.  */
  fd1 = atol (fd1s);
  fd2 = atol (fd2s);

  /* Sanity check.  */
  if (fd1 == fd2)
    error (EXIT_FAILURE, 0, "value of fd1 and fd2 is the same");

  /* First the easy part: read from the file descriptor which is
     supposed to be open.  */
  if (lseek (fd2, 0, SEEK_CUR) != strlen (fd2string))
    error (EXIT_FAILURE, errno, "file 2 not in right position");
  if (lseek (fd2, 0, SEEK_SET) != 0)
    error (EXIT_FAILURE, 0, "cannot reset position in file 2");
  if (read (fd2, buf, sizeof buf) != strlen (fd2string))
    error (EXIT_FAILURE, 0, "cannot read file 2");
  if (memcmp (fd2string, buf, strlen (fd2string)) != 0)
    error (EXIT_FAILURE, 0, "file 2 does not match");

  /* No try to read the first file.  First make sure it is not opened.  */
  if (lseek (fd1, 0, SEEK_CUR) != (off_t) -1 || errno != EBADF)
    error (EXIT_FAILURE, 0, "file 1 (%d) is not closed", fd1);

  /* Now open the file and read it.  */
  fd1 = open (name, O_RDONLY);
  if (fd1 == -1)
    error (EXIT_FAILURE, errno,
	   "cannot open first file \"%s\" for verification", name);

  if (read (fd1, buf, sizeof buf) != strlen (fd1string))
    error (EXIT_FAILURE, errno, "cannot read file 1");
  if (memcmp (fd1string, buf, strlen (fd1string)) != 0)
    error (EXIT_FAILURE, 0, "file 1 does not match");

  return 0;
}


int
do_test (int argc, char *argv[])
{
  pid_t pid;
  int flags;
  int status;

  /* We must have
     - one or four parameters left if called initially
       + path for ld.so		optional
       + "--library-path"	optional
       + the library path	optional
       + the application name
     - three parameters left if called through re-execution
       + file descriptor number which is supposed to be closed
       + the open file descriptor
       + the name of the closed desriptor
  */

  if (restart)
    {
      if (argc != 4)
	error (EXIT_FAILURE, 0, "wrong number of arguments (%d)", argc);

      return handle_restart (argv[1], argv[2], argv[3]);
    }

  if (argc != 2 && argc != 5)
    error (EXIT_FAILURE, 0, "wrong number of arguments (%d)", argc);

  /* Prepare the test.  We are creating two files: one which file descriptor
     will be marked with FD_CLOEXEC, another which is not.  */

   /* Set the bit.  */
   flags = fcntl (temp_fd1, F_GETFD, 0);
   if (flags < 0)
     error (EXIT_FAILURE, errno, "cannot get flags");
   flags |= FD_CLOEXEC;
   if (fcntl (temp_fd1, F_SETFD, flags) < 0)
     error (EXIT_FAILURE, errno, "cannot set flags");

   /* Write something in the files.  */
   if (write (temp_fd1, fd1string, strlen (fd1string)) != strlen (fd1string))
     error (EXIT_FAILURE, errno, "cannot write to first file");
   if (write (temp_fd2, fd2string, strlen (fd2string)) != strlen (fd2string))
     error (EXIT_FAILURE, errno, "cannot write to second file");

  /* We want to test the `exec' function.  To do this we restart the program
     with an additional parameter.  But first create another process.  */
  pid = fork ();
  if (pid == 0)
    {
      char fd1name[18];
      char fd2name[18];

      snprintf (fd1name, sizeof fd1name, "%d", temp_fd1);
      snprintf (fd2name, sizeof fd2name, "%d", temp_fd2);

      /* This is the child.  Construct the command line.  */
      if (argc == 5)
	execl (argv[1], argv[1], argv[2], argv[3], argv[4], "--direct",
	       "--restart", fd1name, fd2name, name1, NULL);
      else
	execl (argv[1], argv[1], "--direct",
	       "--restart", fd1name, fd2name, name1, NULL);

      error (EXIT_FAILURE, errno, "cannot exec");
    }
  else if (pid == (pid_t) -1)
    error (EXIT_FAILURE, errno, "cannot fork");

  /* Wait for the child.  */
  if (waitpid (pid, &status, 0) != pid)
    error (EXIT_FAILURE, errno, "wrong child");

  if (WTERMSIG (status) != 0)
    error (EXIT_FAILURE, 0, "Child terminated incorrectly");
  status = WEXITSTATUS (status);

  return status;
}
