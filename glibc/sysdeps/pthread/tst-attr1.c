/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int
do_test (void)
{
  int i;
  pthread_attr_t a;

  if (pthread_attr_init (&a) != 0)
    {
      puts ("attr_init failed");
      exit (1);
    }

  pthread_mutexattr_t ma;

  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("mutexattr_init failed");
      exit (1);
    }

  pthread_rwlockattr_t rwa;

  if (pthread_rwlockattr_init (&rwa) != 0)
    {
      puts ("rwlockattr_init failed");
      exit (1);
    }

  /* XXX Remove if default value is clear.  */
  pthread_attr_setinheritsched (&a, PTHREAD_INHERIT_SCHED);
  pthread_attr_setschedpolicy (&a, SCHED_OTHER);
  pthread_attr_setscope (&a, PTHREAD_SCOPE_SYSTEM);

  for (i = 0; i < 10000; ++i)
    {
      long int r = random ();

      if (r != PTHREAD_CREATE_DETACHED && r != PTHREAD_CREATE_JOINABLE)
	{
	  int e = pthread_attr_setdetachstate (&a, r);

	  if (e == 0)
	    {
	      printf ("attr_setdetachstate with value %ld succeeded\n", r);
	      exit (1);
	    }
	  if (e != EINVAL)
	    {
	      puts ("attr_setdetachstate didn't return EINVAL");
	      exit (1);
	    }

	  int s;
	  if (pthread_attr_getdetachstate (&a, &s) != 0)
	    {
	      puts ("attr_getdetachstate failed");
	      exit (1);
	    }

	  if (s != PTHREAD_CREATE_JOINABLE)
	    {
	      printf ("\
detach state changed to %d by invalid setdetachstate call\n", s);
	      exit (1);
	    }
	}

      if (r != PTHREAD_INHERIT_SCHED && r != PTHREAD_EXPLICIT_SCHED)
	{
	  int e = pthread_attr_setinheritsched (&a, r);

	  if (e == 0)
	    {
	      printf ("attr_setinheritsched with value %ld succeeded\n", r);
	      exit (1);
	    }
	  if (e != EINVAL)
	    {
	      puts ("attr_setinheritsched didn't return EINVAL");
	      exit (1);
	    }

	  int s;
	  if (pthread_attr_getinheritsched (&a, &s) != 0)
	    {
	      puts ("attr_getinheritsched failed");
	      exit (1);
	    }

	  if (s != PTHREAD_INHERIT_SCHED)
	    {
	      printf ("\
inheritsched changed to %d by invalid setinheritsched call\n", s);
	      exit (1);
	    }
	}

      if (r != SCHED_OTHER && r != SCHED_RR && r != SCHED_FIFO)
	{
	  int e = pthread_attr_setschedpolicy (&a, r);

	  if (e == 0)
	    {
	      printf ("attr_setschedpolicy with value %ld succeeded\n", r);
	      exit (1);
	    }
	  if (e != EINVAL)
	    {
	      puts ("attr_setschedpolicy didn't return EINVAL");
	      exit (1);
	    }

	  int s;
	  if (pthread_attr_getschedpolicy (&a, &s) != 0)
	    {
	      puts ("attr_getschedpolicy failed");
	      exit (1);
	    }

	  if (s != SCHED_OTHER)
	    {
	      printf ("\
schedpolicy changed to %d by invalid setschedpolicy call\n", s);
	      exit (1);
	    }
	}

      if (r != PTHREAD_SCOPE_SYSTEM && r != PTHREAD_SCOPE_PROCESS)
	{
	  int e = pthread_attr_setscope (&a, r);

	  if (e == 0)
	    {
	      printf ("attr_setscope with value %ld succeeded\n", r);
	      exit (1);
	    }
	  if (e != EINVAL)
	    {
	      puts ("attr_setscope didn't return EINVAL");
	      exit (1);
	    }

	  int s;
	  if (pthread_attr_getscope (&a, &s) != 0)
	    {
	      puts ("attr_getscope failed");
	      exit (1);
	    }

	  if (s != PTHREAD_SCOPE_SYSTEM)
	    {
	      printf ("\
contentionscope changed to %d by invalid setscope call\n", s);
	      exit (1);
	    }
	}

      if (r != PTHREAD_PROCESS_PRIVATE && r != PTHREAD_PROCESS_SHARED)
	{
	  int e = pthread_mutexattr_setpshared (&ma, r);

	  if (e == 0)
	    {
	      printf ("mutexattr_setpshared with value %ld succeeded\n", r);
	      exit (1);
	    }
	  if (e != EINVAL)
	    {
	      puts ("mutexattr_setpshared didn't return EINVAL");
	      exit (1);
	    }

	  int s;
	  if (pthread_mutexattr_getpshared (&ma, &s) != 0)
	    {
	      puts ("mutexattr_getpshared failed");
	      exit (1);
	    }

	  if (s != PTHREAD_PROCESS_PRIVATE)
	    {
	      printf ("\
pshared changed to %d by invalid mutexattr_setpshared call\n", s);
	      exit (1);
	    }

	  e = pthread_rwlockattr_setpshared (&rwa, r);

	  if (e == 0)
	    {
	      printf ("rwlockattr_setpshared with value %ld succeeded\n", r);
	      exit (1);
	    }
	  if (e != EINVAL)
	    {
	      puts ("rwlockattr_setpshared didn't return EINVAL");
	      exit (1);
	    }

	  if (pthread_rwlockattr_getpshared (&rwa, &s) != 0)
	    {
	      puts ("rwlockattr_getpshared failed");
	      exit (1);
	    }

	  if (s != PTHREAD_PROCESS_PRIVATE)
	    {
	      printf ("\
pshared changed to %d by invalid rwlockattr_setpshared call\n", s);
	      exit (1);
	    }
	}

      if (r != PTHREAD_CANCEL_ENABLE && r != PTHREAD_CANCEL_DISABLE)
	{
	  int e = pthread_setcancelstate (r, NULL);

	  if (e == 0)
	    {
	      printf ("setcancelstate with value %ld succeeded\n", r);
	      exit (1);
	    }

	  if (e != EINVAL)
	    {
	      puts ("setcancelstate didn't return EINVAL");
	      exit (1);
	    }

	  int s;
	  if (pthread_setcancelstate (PTHREAD_CANCEL_ENABLE, &s) != 0)
	    {
	      puts ("setcancelstate failed for PTHREAD_CANCEL_ENABLE");
	      exit (1);
	    }

	  if (s != PTHREAD_CANCEL_ENABLE)
	    {
	      puts ("invalid setcancelstate changed state");
	      exit (1);
	    }
	}

      if (r != PTHREAD_CANCEL_DEFERRED && r != PTHREAD_CANCEL_ASYNCHRONOUS)
	{
	  int e = pthread_setcanceltype (r, NULL);

	  if (e == 0)
	    {
	      printf ("setcanceltype with value %ld succeeded\n", r);
	      exit (1);
	    }

	  if (e != EINVAL)
	    {
	      puts ("setcanceltype didn't return EINVAL");
	      exit (1);
	    }

	  int s;
	  if (pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &s) != 0)
	    {
	      puts ("setcanceltype failed for PTHREAD_CANCEL_DEFERRED");
	      exit (1);
	    }

	  if (s != PTHREAD_CANCEL_DEFERRED)
	    {
	      puts ("invalid setcanceltype changed state");
	      exit (1);
	    }
	}
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
