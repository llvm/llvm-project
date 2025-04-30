/* BZ #17977 _res_hconf_reorder_addrs test.

   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <pthread.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

static struct timespec ts;

/* The first thread that gets a lock in _res_hconf_reorder_addrs()
   should hold the lock long enough to make two other threads blocked.
   This is achieved by slowing down realloc(3) that is called several times
   by _res_hconf_reorder_addrs().  */

void *
realloc (void *ptr, size_t len)
{
  static void *(*fun) (void *, size_t);

  if (!fun)
    fun = dlsym (RTLD_NEXT, "realloc");

  if (ts.tv_nsec)
    nanosleep (&ts, NULL);

  return (*fun) (ptr, len);
}

static void *
resolve (void *arg)
{
  struct in_addr addr;
  struct hostent ent;
  struct hostent *result;
  int err;
  char buf[1024];

  addr.s_addr = htonl (INADDR_LOOPBACK);
  (void) gethostbyaddr_r ((void *) &addr, sizeof (addr), AF_INET,
		          &ent, buf, sizeof (buf), &result, &err);
  return arg;
}

static int
do_test (void)
{
  #define N 3
  pthread_t thr[N];
  unsigned int i;
  int result = 0;

  /* turn on realloc slowdown */
  ts.tv_nsec = 100000000;

  for (i = 0; i < N; ++i)
    {
      int rc = pthread_create (&thr[i], NULL, resolve, NULL);

      if (rc)
	{
	  printf ("pthread_create: %s\n", strerror(rc));
	  exit (1);
	}
    }

  for (i = 0; i < N; ++i)
    {
      void *retval;
      int rc = pthread_join (thr[i], &retval);

      if (rc)
	{
	  printf ("pthread_join: %s\n", strerror(rc));
	  exit (1);
	}
      if (retval)
	{
	  printf ("thread %u exit status %p\n", i, retval);
	  result = 1;
	}
    }

  /* turn off realloc slowdown, no longer needed */
  ts.tv_nsec = 0;

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
