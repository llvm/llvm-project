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
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>


int *condition;

static int
do_test (void)
{
  size_t ps = sysconf (_SC_PAGESIZE);
  char tmpfname[] = "/tmp/tst-cond6.XXXXXX";
  char data[ps];
  void *mem;
  int fd;
  pthread_mutexattr_t ma;
  pthread_mutex_t *mut1;
  pthread_mutex_t *mut2;
  pthread_condattr_t ca;
  pthread_cond_t *cond;
  pid_t pid;
  int result = 0;

  fd = mkstemp (tmpfname);
  if (fd == -1)
    {
      printf ("cannot open temporary file: %m\n");
      exit (1);
    }

  /* Make sure it is always removed.  */
  unlink (tmpfname);

  /* Create one page of data.  */
  memset (data, '\0', ps);

  /* Write the data to the file.  */
  if (write (fd, data, ps) != (ssize_t) ps)
    {
      puts ("short write");
      exit (1);
    }

  mem = mmap (NULL, ps, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mem == MAP_FAILED)
    {
      printf ("mmap failed: %m\n");
      exit (1);
    }

  mut1 = (pthread_mutex_t *) (((uintptr_t) mem
			       + __alignof (pthread_mutex_t))
			      & ~(__alignof (pthread_mutex_t) - 1));
  mut2 = mut1 + 1;

  cond = (pthread_cond_t *) (((uintptr_t) (mut2 + 1)
			      + __alignof (pthread_cond_t))
			     & ~(__alignof (pthread_cond_t) - 1));

  condition = (int *) (((uintptr_t) (cond + 1) + __alignof (int))
		       & ~(__alignof (int) - 1));

  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("mutexattr_init failed");
      exit (1);
    }

  if (pthread_mutexattr_setpshared (&ma, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("mutexattr_setpshared failed");
      exit (1);
    }

  if (pthread_mutex_init (mut1, &ma) != 0)
    {
      puts ("1st mutex_init failed");
      exit (1);
    }

  if (pthread_mutex_init (mut2, &ma) != 0)
    {
      puts ("2nd mutex_init failed");
      exit (1);
    }

  if (pthread_condattr_init (&ca) != 0)
    {
      puts ("condattr_init failed");
      exit (1);
    }

  if (pthread_condattr_setpshared (&ca, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("condattr_setpshared failed");
      exit (1);
    }

  if (pthread_cond_init (cond, &ca) != 0)
    {
      puts ("cond_init failed");
      exit (1);
    }

  if (pthread_mutex_lock (mut1) != 0)
    {
      puts ("parent: 1st mutex_lock failed");
      exit (1);
    }

  puts ("going to fork now");
  pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      exit (1);
    }
  else if (pid == 0)
    {
      struct timespec ts;
      struct timeval tv;

      if (pthread_mutex_lock (mut2) != 0)
	{
	  puts ("child: mutex_lock failed");
	  exit (1);
	}

      if (pthread_mutex_unlock (mut1) != 0)
	{
	  puts ("child: 1st mutex_unlock failed");
	  exit (1);
	}

      if (gettimeofday (&tv, NULL) != 0)
	{
	  puts ("gettimeofday failed");
	  exit (1);
	}

      TIMEVAL_TO_TIMESPEC (&tv, &ts);
      ts.tv_nsec += 500000000;
      if (ts.tv_nsec >= 1000000000)
	{
	  ts.tv_nsec -= 1000000000;
	  ++ts.tv_sec;
	}

      do
	if (pthread_cond_timedwait (cond, mut2, &ts) != 0)
	  {
	    puts ("child: cond_wait failed");
	    exit (1);
	  }
      while (*condition == 0);

      if (pthread_mutex_unlock (mut2) != 0)
	{
	  puts ("child: 2nd mutex_unlock failed");
	  exit (1);
	}

      puts ("child done");
    }
  else
    {
      int status;

      if (pthread_mutex_lock (mut1) != 0)
	{
	  puts ("parent: 2nd mutex_lock failed");
	  exit (1);
	}

      if (pthread_mutex_lock (mut2) != 0)
	{
	  puts ("parent: 3rd mutex_lock failed");
	  exit (1);
	}

      if (pthread_cond_signal (cond) != 0)
	{
	  puts ("parent: cond_signal failed");
	  exit (1);
	}

      *condition = 1;

      if (pthread_mutex_unlock (mut2) != 0)
	{
	  puts ("parent: mutex_unlock failed");
	  exit (1);
	}

      puts ("waiting for child");

      waitpid (pid, &status, 0);
      result |= status;

      puts ("parent done");
    }

 return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
