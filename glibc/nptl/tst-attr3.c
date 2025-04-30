/* pthread_getattr_np test.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <stackinfo.h>
#include <libc-diag.h>

static void *
tf (void *arg)
{
  pthread_attr_t a, *ap, a2;
  int err;
  void *result = NULL;

  if (arg == NULL)
    {
      ap = &a2;
      err = pthread_attr_init (ap);
      if (err)
	{
	  error (0, err, "pthread_attr_init failed");
	  return tf;
	}
    }
  else
    ap = (pthread_attr_t *) arg;

  err = pthread_getattr_np (pthread_self (), &a);
  if (err)
    {
      error (0, err, "pthread_getattr_np failed");
      result = tf;
    }

  int detachstate1, detachstate2;
  err = pthread_attr_getdetachstate (&a, &detachstate1);
  if (err)
    {
      error (0, err, "pthread_attr_getdetachstate failed");
      result = tf;
    }
  else
    {
      err = pthread_attr_getdetachstate (ap, &detachstate2);
      if (err)
	{
	  error (0, err, "pthread_attr_getdetachstate failed");
	  result = tf;
	}
      else if (detachstate1 != detachstate2)
	{
	  error (0, 0, "detachstate differs %d != %d",
		 detachstate1, detachstate2);
	  result = tf;
	}
    }

  void *stackaddr;
  size_t stacksize;
  err = pthread_attr_getstack (&a, &stackaddr, &stacksize);
  if (err)
    {
      error (0, err, "pthread_attr_getstack failed");
      result = tf;
    }
  else if ((void *) &a < stackaddr
	   || (void *) &a >= stackaddr + stacksize)
    {
      error (0, 0, "pthread_attr_getstack returned range does not cover thread's stack");
      result = tf;
    }
  else
    printf ("thread stack %p-%p (0x%zx)\n", stackaddr, stackaddr + stacksize,
	    stacksize);

  size_t guardsize1, guardsize2;
  err = pthread_attr_getguardsize (&a, &guardsize1);
  if (err)
    {
      error (0, err, "pthread_attr_getguardsize failed");
      result = tf;
    }
  else
    {
      err = pthread_attr_getguardsize (ap, &guardsize2);
      if (err)
	{
	  error (0, err, "pthread_attr_getguardsize failed");
	  result = tf;
	}
      else if (guardsize1 != guardsize2)
	{
	  error (0, 0, "guardsize differs %zd != %zd",
		 guardsize1, guardsize2);
	  result = tf;
	}
      else
	printf ("thread guardsize %zd\n", guardsize1);
    }

  int scope1, scope2;
  err = pthread_attr_getscope (&a, &scope1);
  if (err)
    {
      error (0, err, "pthread_attr_getscope failed");
      result = tf;
    }
  else
    {
      err = pthread_attr_getscope (ap, &scope2);
      if (err)
	{
	  error (0, err, "pthread_attr_getscope failed");
	  result = tf;
	}
      else if (scope1 != scope2)
	{
	  error (0, 0, "scope differs %d != %d",
		 scope1, scope2);
	  result = tf;
	}
    }

  int inheritsched1, inheritsched2;
  err = pthread_attr_getinheritsched (&a, &inheritsched1);
  if (err)
    {
      error (0, err, "pthread_attr_getinheritsched failed");
      result = tf;
    }
  else
    {
      err = pthread_attr_getinheritsched (ap, &inheritsched2);
      if (err)
	{
	  error (0, err, "pthread_attr_getinheritsched failed");
	  result = tf;
	}
      else if (inheritsched1 != inheritsched2)
	{
	  error (0, 0, "inheritsched differs %d != %d",
		 inheritsched1, inheritsched2);
	  result = tf;
	}
    }

  cpu_set_t c1, c2;
  err = pthread_getaffinity_np (pthread_self (), sizeof (c1), &c1);
  if (err == 0)
    {
      err = pthread_attr_getaffinity_np (&a, sizeof (c2), &c2);
      if (err)
	{
	  error (0, err, "pthread_attr_getaffinity_np failed");
	  result = tf;
	}
      else if (memcmp (&c1, &c2, sizeof (c1)))
	{
	  error (0, 0, "pthread_attr_getaffinity_np returned different CPU mask than pthread_getattr_np");
	  result = tf;
	}
    }

  err = pthread_attr_destroy (&a);
  if (err)
    {
      error (0, err, "pthread_attr_destroy failed");
      result = tf;
    }

  if (ap == &a2)
    {
      err = pthread_attr_destroy (ap);
      if (err)
	{
	  error (0, err, "pthread_attr_destroy failed");
	  result = tf;
	}
    }

  return result;
}


static int
do_test (void)
{
  int result = 0;
  pthread_attr_t a;
  cpu_set_t c1, c2;

  int err = pthread_attr_init (&a);
  if (err)
    {
      error (0, err, "pthread_attr_init failed");
      result = 1;
    }

  err = pthread_attr_getaffinity_np (&a, sizeof (c1), &c1);
  if (err && err != ENOSYS)
    {
      error (0, err, "pthread_attr_getaffinity_np failed");
      result = 1;
    }

  err = pthread_attr_destroy (&a);
  if (err)
    {
      error (0, err, "pthread_attr_destroy failed");
      result = 1;
    }

  err = pthread_getattr_np (pthread_self (), &a);
  if (err)
    {
      error (0, err, "pthread_getattr_np failed");
      result = 1;
    }

  int detachstate;
  err = pthread_attr_getdetachstate (&a, &detachstate);
  if (err)
    {
      error (0, err, "pthread_attr_getdetachstate failed");
      result = 1;
    }
  else if (detachstate != PTHREAD_CREATE_JOINABLE)
    {
      error (0, 0, "initial thread not joinable");
      result = 1;
    }

  void *stackaddr;
  size_t stacksize;
  err = pthread_attr_getstack (&a, &stackaddr, &stacksize);
  if (err)
    {
      error (0, err, "pthread_attr_getstack failed");
      result = 1;
    }
  else if ((void *) &a < stackaddr
	   || (void *) &a >= stackaddr + stacksize)
    {
      error (0, 0, "pthread_attr_getstack returned range does not cover main's stack");
      result = 1;
    }
  else
    printf ("initial thread stack %p-%p (0x%zx)\n", stackaddr,
	    stackaddr + stacksize, stacksize);

  size_t guardsize;
  err = pthread_attr_getguardsize (&a, &guardsize);
  if (err)
    {
      error (0, err, "pthread_attr_getguardsize failed");
      result = 1;
    }
  else if (guardsize != 0)
    {
      error (0, 0, "pthread_attr_getguardsize returned %zd != 0",
	     guardsize);
      result = 1;
    }

  int scope;
  err = pthread_attr_getscope (&a, &scope);
  if (err)
    {
      error (0, err, "pthread_attr_getscope failed");
      result = 1;
    }
  else if (scope != PTHREAD_SCOPE_SYSTEM)
    {
      error (0, 0, "pthread_attr_getscope returned %d != PTHREAD_SCOPE_SYSTEM",
	     scope);
      result = 1;
    }

  int inheritsched;
  err = pthread_attr_getinheritsched (&a, &inheritsched);
  if (err)
    {
      error (0, err, "pthread_attr_getinheritsched failed");
      result = 1;
    }
  else if (inheritsched != PTHREAD_INHERIT_SCHED)
    {
      error (0, 0, "pthread_attr_getinheritsched returned %d != PTHREAD_INHERIT_SCHED",
	     inheritsched);
      result = 1;
    }

  err = pthread_getaffinity_np (pthread_self (), sizeof (c1), &c1);
  if (err == 0)
    {
      err = pthread_attr_getaffinity_np (&a, sizeof (c2), &c2);
      if (err)
	{
	  error (0, err, "pthread_attr_getaffinity_np failed");
	  result = 1;
	}
      else if (memcmp (&c1, &c2, sizeof (c1)))
	{
	  error (0, 0, "pthread_attr_getaffinity_np returned different CPU mask than pthread_getattr_np");
	  result = 1;
	}
    }

  err = pthread_attr_destroy (&a);
  if (err)
    {
      error (0, err, "pthread_attr_destroy failed");
      result = 1;
    }

  pthread_t th;
  err = pthread_create (&th, NULL, tf, NULL);
  if (err)
    {
      error (0, err, "pthread_create #1 failed");
      result = 1;
    }
  else
    {
      void *ret;
      err = pthread_join (th, &ret);
      if (err)
	{
	  error (0, err, "pthread_join #1 failed");
	  result = 1;
	}
      else if (ret != NULL)
	result = 1;
    }

  err = pthread_attr_init (&a);
  if (err)
    {
      error (0, err, "pthread_attr_init failed");
      result = 1;
    }

  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* GCC 8 warns about aliasing of the restrict-qualified arguments
     passed &a.  Since pthread_create does not dereference its fourth
     argument, this aliasing, which is deliberate in this test, cannot
     in fact cause problems.  */
  DIAG_IGNORE_NEEDS_COMMENT (8, "-Wrestrict");
#endif
  err = pthread_create (&th, &a, tf, &a);
  DIAG_POP_NEEDS_COMMENT;
  if (err)
    {
      error (0, err, "pthread_create #2 failed");
      result = 1;
    }
  else
    {
      void *ret;
      err = pthread_join (th, &ret);
      if (err)
	{
	  error (0, err, "pthread_join #2 failed");
	  result = 1;
	}
      else if (ret != NULL)
	result = 1;
    }

  err = pthread_attr_setguardsize (&a, 16 * sysconf (_SC_PAGESIZE));
  if (err)
    {
      error (0, err, "pthread_attr_setguardsize failed");
      result = 1;
    }

  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* GCC 8 warns about aliasing of the restrict-qualified arguments
     passed &a.  Since pthread_create does not dereference its fourth
     argument, this aliasing, which is deliberate in this test, cannot
     in fact cause problems.  */
  DIAG_IGNORE_NEEDS_COMMENT (8, "-Wrestrict");
#endif
  err = pthread_create (&th, &a, tf, &a);
  DIAG_POP_NEEDS_COMMENT;
  if (err)
    {
      error (0, err, "pthread_create #3 failed");
      result = 1;
    }
  else
    {
      void *ret;
      err = pthread_join (th, &ret);
      if (err)
	{
	  error (0, err, "pthread_join #3 failed");
	  result = 1;
	}
      else if (ret != NULL)
	result = 1;
    }

  err = pthread_attr_destroy (&a);
  if (err)
    {
      error (0, err, "pthread_attr_destroy failed");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
