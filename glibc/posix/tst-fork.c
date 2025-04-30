/* Tests for fork.
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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wait.h>


static const char testdata[] = "This is a test";
static const char testdata2[] = "And here we go again";


int
main (void)
{
  const char *tmpdir = getenv ("TMPDIR");
  char buf[100];
  size_t tmpdirlen;
  char *name;
  int fd;
  pid_t pid;
  pid_t ppid;
  off_t off;
  int status;

  if (tmpdir == NULL || *tmpdir == '\0')
    tmpdir = "/tmp";
  tmpdirlen = strlen (tmpdir);

  name = (char *) malloc (tmpdirlen + strlen ("/forkXXXXXX") + 1);
  if (name == NULL)
    error (EXIT_FAILURE, errno, "cannot allocate file name");

  mempcpy (mempcpy (name, tmpdir, tmpdirlen),
	   "/forkXXXXXX", sizeof ("/forkXXXXXX"));

  /* Open our test file.   */
  fd = mkstemp (name);
  if (fd == -1)
     error (EXIT_FAILURE, errno, "cannot open test file `%s'", name);

  /* Make sure it gets removed.  */
  unlink (name);

  /* Write some data.  */
  if (write (fd, testdata, strlen (testdata)) != strlen (testdata))
    error (EXIT_FAILURE, errno, "cannot write test data");

  /* Get the position in the stream.  */
  off = lseek (fd, 0, SEEK_CUR);
  if (off == (off_t) -1 || off != strlen (testdata))
    error (EXIT_FAILURE, errno, "wrong file position");

  /* Get the parent PID.  */
  ppid = getpid ();

  /* Now fork of the process.  */
  pid = fork ();
  if (pid == 0)
    {
      /* One little test first: the PID must have changed.  */
      if (getpid () == ppid)
	error (EXIT_FAILURE, 0, "child and parent have same PID");

      /* Test the `getppid' function.  */
      pid = getppid ();
      if (pid == (pid_t) -1 ? errno != ENOSYS : pid != ppid)
	error (EXIT_FAILURE, 0,
	       "getppid returned wrong PID (%ld, should be %ld)",
	       (long int) pid, (long int) ppid);

      /* This is the child.  First get the position of the descriptor.  */
      off = lseek (fd, 0, SEEK_CUR);
      if (off == (off_t) -1 || off != strlen (testdata))
	error (EXIT_FAILURE, errno, "wrong file position in child");

      /* Reset the position.  */
      if (lseek (fd, 0, SEEK_SET) != 0)
	error (EXIT_FAILURE, errno, "cannot reset position in child");

      /* Read the data.  */
      if (read (fd, buf, sizeof buf) != strlen (testdata))
	error (EXIT_FAILURE, errno, "cannot read data in child");

      /* Compare the data.  */
      if (memcmp (buf, testdata, strlen (testdata)) != 0)
	error (EXIT_FAILURE, 0, "data comparison failed in child");

      /* Reset position again.  */
      if (lseek (fd, 0, SEEK_SET) != 0)
	error (EXIT_FAILURE, errno, "cannot reset position again in child");

      /* Write new data.  */
      if (write (fd, testdata2, strlen (testdata2)) != strlen (testdata2))
	error (EXIT_FAILURE, errno, "cannot write new data in child");

      /* Close the file.  This must not remove it.  */
      close (fd);

      _exit (0);
    }
  else if (pid < 0)
    /* Something went wrong.  */
    error (EXIT_FAILURE, errno, "cannot fork");

  /* Wait for the child.  */
  if (waitpid (pid, &status, 0) != pid)
    error (EXIT_FAILURE, 0, "Oops, wrong test program terminated");

  if (WTERMSIG (status) != 0)
    error (EXIT_FAILURE, 0, "Child terminated incorrectly");
  status = WEXITSTATUS (status);

  if (status == 0)
    {
      /* Test whether the child wrote the right data.  First test the
	 position.  It must be the same as in the child.  */
      if (lseek (fd, 0, SEEK_CUR) != strlen (testdata2))
	error (EXIT_FAILURE, 0, "file position not changed");

      if (lseek (fd, 0, SEEK_SET) != 0)
	error (EXIT_FAILURE, errno, "cannot reset file position");

      if (read (fd, buf, sizeof buf) != strlen (testdata2))
	error (EXIT_FAILURE, errno, "cannot read new data");

      if (memcmp (buf, testdata2, strlen (testdata2)) != 0)
	error (EXIT_FAILURE, 0, "new data not read correctly");
    }

  return status;
}
