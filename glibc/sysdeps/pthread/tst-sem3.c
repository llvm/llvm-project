/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>


int
do_test (void)
{
  size_t ps = sysconf (_SC_PAGESIZE);
  char tmpfname[] = "/tmp/tst-sem3.XXXXXX";
  char data[ps];
  void *mem;
  int fd;
  sem_t *s;
  pid_t pid;
  char *p;

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

  s = (sem_t *) (((uintptr_t) mem + __alignof (sem_t))
		 & ~(__alignof (sem_t) - 1));
  p = (char *) (s + 1);

  if (sem_init (s, 1, 1) == -1)
    {
      puts ("init failed");
      return 1;
    }

  if (TEMP_FAILURE_RETRY (sem_wait (s)) == -1)
    {
      puts ("1st wait failed");
      return 1;
    }

  errno = 0;
  if (TEMP_FAILURE_RETRY (sem_trywait (s)) != -1)
    {
      puts ("trywait succeeded");
      return 1;
    }
  else if (errno != EAGAIN)
    {
      puts ("trywait didn't return EAGAIN");
      return 1;
    }

  *p = 0;

  puts ("going to fork now");
  pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      return 1;
    }
  else if (pid == 0)
    {
      /* Play some lock ping-pong.  It's our turn to unlock first.  */
      if ((*p)++ != 0)
	{
	  puts ("child: *p != 0");
	  return 1;
	}

      if (sem_post (s) == -1)
	{
	  puts ("child: 1st post failed");
	  return 1;
	}

      puts ("child done");
    }
  else
    {
      if (TEMP_FAILURE_RETRY (sem_wait (s)) == -1)
	{
	  printf ("parent: 2nd wait failed: %m\n");
	  return 1;
	}

      if (*p != 1)
	{
	  puts ("*p != 1");
	  return 1;
	}

      puts ("parent done");
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
