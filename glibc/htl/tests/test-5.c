/* Test signals.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#define _GNU_SOURCE

#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <error.h>
#include <assert.h>
#include <sys/resource.h>
#include <sys/wait.h>

void *
thr (void *arg)
{
  *(int *) 0 = 0;
  return 0;
}

int foobar;

int
main (int argc, char *argv[])
{
  error_t err;
  pid_t child;

  struct rlimit limit;

  limit.rlim_cur = 0;
  limit.rlim_max = 0;

  err = setrlimit (RLIMIT_CORE, &limit);
  if (err)
    error (1, err, "setrlimit");

  child = fork ();
  switch (child)
    {
    case -1:
      error (1, errno, "fork");
      break;

    case 0:
      {
	pthread_t tid;
	void *ret;

	err = pthread_create (&tid, 0, thr, 0);
	if (err)
	  error (1, err, "pthread_create");

	err = pthread_join (tid, &ret);
	assert_perror (err);

	/* Should have never returned.  Our parent expects us to fail
	   thus we succeed and indicate the error.  */
	return 0;
      }

    default:
      {
	pid_t pid;
	int status;

	pid = waitpid (child, &status, 0);
	printf ("pid = %d; child = %d; status = %d\n", pid, child, status);
	assert (pid == child);
	assert (status != 0);
      }
    }

  return 0;
}
