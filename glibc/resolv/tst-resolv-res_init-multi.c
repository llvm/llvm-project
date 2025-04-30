/* Multi-threaded test for resolver initialization.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <netdb.h>
#include <resolv.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xthread.h>

/* Whether name lookups succeed does not really matter.  We use this
   to trigger initialization of the resolver.  */
static const char *test_hostname = "www.gnu.org";

/* The different initialization methods.  */
enum test_type { init, byname, gai };
enum { type_count = 3 };

/* Thread function.  Perform a few resolver options.  */
static void *
thread_func (void *closure)
{
  enum test_type *ptype = closure;
  /* Perform a few calls to the requested operation.  */
  TEST_VERIFY (*ptype >= 0);
  TEST_VERIFY (*ptype < (int) type_count);
  for (int i = 0; i < 3; ++i)
    switch (*ptype)
      {
      case init:
	res_init ();
	break;
      case byname:
	gethostbyname (test_hostname);
	break;
      case gai:
	{
	  struct addrinfo hints = { 0, };
	  struct addrinfo *ai = NULL;
	  if (getaddrinfo (test_hostname, "80", &hints, &ai) == 0)
	    freeaddrinfo (ai);
	}
	break;
      }
  free (ptype);
  return NULL;
}

static int
do_test (void)
{
  /* Start a small number of threads which perform resolver
     operations.  */
  enum { thread_count = 30 };

  pthread_t threads[thread_count];
  for (int i = 0; i < thread_count; ++i)
    {
      enum test_type *ptype = xmalloc (sizeof (*ptype));
      *ptype = i % type_count;
      threads[i] = xpthread_create (NULL, thread_func, ptype);
    }
  for (int i = 0; i < type_count; ++i)
    {
      enum test_type *ptype = xmalloc (sizeof (*ptype));
      *ptype = i;
      thread_func (ptype);
    }
  for (int i = 0; i < thread_count; ++i)
    xpthread_join (threads[i]);
  return 0;
}

#include <support/test-driver.c>
