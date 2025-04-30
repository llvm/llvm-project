/* Test the stack guard.
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
#include <assert.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

size_t stacksize;

void *
thr (void *arg)
{
  int i;
  char *foo;

  foo = alloca (3 * stacksize / 4);
  for (i = 0; i < sizeof foo; i++)
    foo[i] = -1;

  return (void *) 1;
}

int
main (int argc, char *argv[])
{
  error_t err;
  pid_t child;

  child = fork ();
  switch (child)
    {
    case -1:
      error (1, errno, "fork");
      break;

    case 0:
      {
	pthread_attr_t attr;
	pthread_t tid;
	void *ret;

	err = pthread_attr_init (&attr);
	assert_perror (err);

	err = pthread_attr_getstacksize (&attr, &stacksize);
	assert_perror (err);

	err = pthread_attr_setguardsize (&attr, stacksize / 2);
	if (err == ENOTSUP)
	  {
	    printf ("Stack guard attribute not supported.\n");
	    return 1;
	  }
	assert_perror (err);

	err = pthread_create (&tid, &attr, thr, 0);
	assert_perror (err);

	err = pthread_attr_destroy (&attr);
	assert_perror (err);

	err = pthread_join (tid, &ret);
	/* Should never be successful.  */
	printf ("Thread did not segfault!?!\n");
	assert_perror (err);
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
