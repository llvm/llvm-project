/* Tests process-shared barriers.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>


static int
do_test (void)
{
  size_t ps = sysconf (_SC_PAGESIZE);
  char tmpfname[] = "/tmp/tst-barrier2.XXXXXX";
  char data[ps];
  void *mem;
  int fd;
  pthread_barrier_t *b;
  pthread_barrierattr_t a;
  pid_t pid;
  int serials = 0;
  int cnt;
  int status;
  int p;

  fd = mkstemp (tmpfname);
  if (fd == -1)
    {
      printf ("cannot open temporary file: %m\n");
      return 1;
    }

  /* Make sure it is always removed.  */
  unlink (tmpfname);

  /* Create one page of data.  */
  memset (data, '\0', ps);

  /* Write the data to the file.  */
  if (write (fd, data, ps) != (ssize_t) ps)
    {
      puts ("short write");
      return 1;
    }

  mem = mmap (NULL, ps, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mem == MAP_FAILED)
    {
      printf ("mmap failed: %m\n");
      return 1;
    }

  b = (pthread_barrier_t *) (((uintptr_t) mem + __alignof (pthread_barrier_t))
			     & ~(__alignof (pthread_barrier_t) - 1));

  if (pthread_barrierattr_init (&a) != 0)
    {
      puts ("barrierattr_init failed");
      return 1;
    }

  if (pthread_barrierattr_getpshared (&a, &p) != 0)
    {
      puts ("1st barrierattr_getpshared failed");
      return 1;
    }

  if (p != PTHREAD_PROCESS_PRIVATE)
    {
      puts ("default pshared value wrong");
      return 1;
    }

  if (pthread_barrierattr_setpshared (&a, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("barrierattr_setpshared failed");
      return 1;
    }

  if (pthread_barrierattr_getpshared (&a, &p) != 0)
    {
      puts ("2nd barrierattr_getpshared failed");
      return 1;
    }

  if (p != PTHREAD_PROCESS_SHARED)
    {
      puts ("pshared value after setpshared call wrong");
      return 1;
    }

  if (pthread_barrier_init (b, &a, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_barrierattr_destroy (&a) != 0)
    {
      puts ("barrierattr_destroy failed");
      return 1;
    }

  puts ("going to fork now");
  pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      return 1;
    }

  /* Just to be sure we don't hang forever.  */
  alarm (4);

#define N 30
  for (cnt = 0; cnt < N; ++cnt)
    {
      int e;

      e = pthread_barrier_wait (b);
      if (e == PTHREAD_BARRIER_SERIAL_THREAD)
	++serials;
      else if (e != 0)
	{
	  printf ("%s: barrier_wait returned value %d != 0 and PTHREAD_BARRIER_SERIAL_THREAD\n",
		  pid == 0 ? "child" : "parent", e);
	  return 1;
	}
    }

  alarm (0);

  printf ("%s: was %d times the serial thread\n",
	  pid == 0 ? "child" : "parent", serials);

  if (pid == 0)
    /* The child.  Pass the number of times we had the serializing
       thread back to the parent.  */
    exit (serials);

  if (waitpid (pid, &status, 0) != pid)
    {
      puts ("waitpid failed");
      return 1;
    }

  if (!WIFEXITED (status))
    {
      puts ("child exited abnormally");
      return 1;
    }

  if (WEXITSTATUS (status) + serials != N)
    {
      printf ("total number of serials is %d, expected %d\n",
	      WEXITSTATUS (status) + serials, N);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
