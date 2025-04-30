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
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>


static int
do_test (void)
{
  size_t ps = sysconf (_SC_PAGESIZE);
  char tmpfname[] = "/tmp/tst-rwlock12.XXXXXX";
  char data[ps];
  void *mem;
  int fd;
  pthread_mutex_t *m;
  pthread_mutexattr_t ma;
  pthread_rwlock_t *r;
  pthread_rwlockattr_t ra;
  pid_t pid;
  int status = 0;

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

  r = (pthread_rwlock_t *) (((uintptr_t) mem + __alignof (pthread_rwlock_t))
			    & ~(__alignof (pthread_rwlock_t) - 1));
  /* The following assumes alignment for a mutex is at least as high
     as that for a rwlock.  Which is true in our case.  */
  m = (pthread_mutex_t *) (r + 1);

  if (pthread_rwlockattr_init (&ra) != 0)
    {
      puts ("rwlockattr_init failed");
      return 1;
    }

  if (pthread_rwlockattr_setpshared (&ra, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("rwlockattr_setpshared failed");
      return 1;
    }

  if (pthread_rwlock_init (r, &ra) != 0)
    {
      puts ("rwlock_init failed");
      return 1;
    }

  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("rwlockattr_init failed");
      return 1;
    }

  if (pthread_mutexattr_setpshared (&ma, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("mutexattr_setpshared failed");
      return 1;
    }

  if (pthread_mutex_init (m, &ma) != 0)
    {
      puts ("mutex_init failed");
      return 1;
    }

  /* Lock the mutex.  */
  if (pthread_mutex_lock (m) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  puts ("going to fork now");
  pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      return 1;
    }
  else if (pid == 0)
    {
      /* Lock the mutex.  */
      if (pthread_mutex_lock (m) != 0)
	{
	  puts ("child: mutex_lock failed");
	  return 1;
	}

      /* Try to get the rwlock.  */
      if (pthread_rwlock_trywrlock (r) == 0)
	{
	  puts ("rwlock_trywrlock succeeded");
	  return 1;
	}

      /* Try again.  */
      struct timespec ts = { .tv_sec = 0, .tv_nsec = 500000000 };
      int e = pthread_rwlock_timedwrlock (r, &ts);
      if (e == 0)
	{
	  puts ("rwlock_timedwrlock succeeded");
	  return 1;
	}
      if (e != ETIMEDOUT)
	{
	  puts ("rwlock_timedwrlock didn't return ETIMEDOUT");
	  status = 1;
	}

      if (pthread_rwlock_tryrdlock (r) == 0)
	{
	  puts ("rwlock_tryrdlock succeeded");
	  return 1;
	}

      e = pthread_rwlock_timedrdlock (r, &ts);
      if (e == 0)
	{
	  puts ("rwlock_timedrdlock succeeded");
	  return 1;
	}
      if (e != ETIMEDOUT)
	{
	  puts ("rwlock_timedrdlock didn't return ETIMEDOUT");
	  status = 1;
	}
    }
  else
    {
      /* Lock the rwlock for writing.  */
      if (pthread_rwlock_wrlock (r) != 0)
	{
	  puts ("rwlock_wrlock failed");
	  kill (pid, SIGTERM);
	  return 1;
	}

      /* Allow the child to run.  */
      if (pthread_mutex_unlock (m) != 0)
	{
	  puts ("mutex_unlock failed");
	  kill (pid, SIGTERM);
	  return 1;
	}

      /* Just wait for the child.  */
      if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
	{
	  puts ("waitpid failed");
	  kill (pid, SIGTERM);
	  return 1;
	}
    }

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
