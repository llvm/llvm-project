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
  pthread_attr_t a;

  if (pthread_attr_init (&a) != 0)
    {
      puts ("attr_init failed");
      exit (1);
    }

  /* Check default value of detach state.  */
  int s;
  if (pthread_attr_getdetachstate (&a, &s) != 0)
    {
      puts ("1st attr_getdestachstate failed");
      exit (1);
    }
  if (s != PTHREAD_CREATE_JOINABLE)
    {
      printf ("\
default detach state wrong: %d, expected %d (PTHREAD_CREATE_JOINABLE)\n",
	      s, PTHREAD_CREATE_JOINABLE);
      exit (1);
    }

  int e = pthread_attr_setdetachstate (&a, PTHREAD_CREATE_DETACHED);
  if (e != 0)
    {
      puts ("1st attr_setdetachstate failed");
      exit (1);
    }
  if (pthread_attr_getdetachstate (&a, &s) != 0)
    {
      puts ("2nd attr_getdestachstate failed");
      exit (1);
    }
  if (s != PTHREAD_CREATE_DETACHED)
    {
      puts ("PTHREAD_CREATE_DETACHED set, but not given back");
      exit (1);
    }

  e = pthread_attr_setdetachstate (&a, PTHREAD_CREATE_JOINABLE);
  if (e != 0)
    {
      puts ("2nd attr_setdetachstate failed");
      exit (1);
    }
  if (pthread_attr_getdetachstate (&a, &s) != 0)
    {
      puts ("3rd attr_getdestachstate failed");
      exit (1);
    }
  if (s != PTHREAD_CREATE_JOINABLE)
    {
      puts ("PTHREAD_CREATE_JOINABLE set, but not given back");
      exit (1);
    }


  size_t g;
  if (pthread_attr_getguardsize (&a, &g) != 0)
    {
      puts ("1st attr_getguardsize failed");
      exit (1);
    }
  if (g != (size_t) sysconf (_SC_PAGESIZE))
    {
      printf ("default guardsize %zu, expected %ld (PAGESIZE)\n",
	      g, sysconf (_SC_PAGESIZE));
      exit (1);
    }

  e = pthread_attr_setguardsize (&a, 0);
  if (e != 0)
    {
      puts ("1st attr_setguardsize failed");
      exit (1);
    }
  if (pthread_attr_getguardsize (&a, &g) != 0)
    {
      puts ("2nd attr_getguardsize failed");
      exit (1);
    }
  if (g != 0)
    {
      printf ("guardsize set to zero but %zu returned\n", g);
      exit (1);
    }

  e = pthread_attr_setguardsize (&a, 1);
  if (e != 0)
    {
      puts ("2nd attr_setguardsize failed");
      exit (1);
    }
  if (pthread_attr_getguardsize (&a, &g) != 0)
    {
      puts ("3rd attr_getguardsize failed");
      exit (1);
    }
  if (g != 1)
    {
      printf ("guardsize set to 1 but %zu returned\n", g);
      exit (1);
    }


  if (pthread_attr_getinheritsched (&a, &s) != 0)
    {
      puts ("1st attr_getinheritsched failed");
      exit (1);
    }
  /* XXX What is the correct default value.  */
  if (s != PTHREAD_INHERIT_SCHED && s != PTHREAD_EXPLICIT_SCHED)
    {
      puts ("incorrect default value for inheritsched");
      exit (1);
    }

  e = pthread_attr_setinheritsched (&a, PTHREAD_EXPLICIT_SCHED);
  if (e != 0)
    {
      puts ("1st attr_setinheritsched failed");
      exit (1);
    }
  if (pthread_attr_getinheritsched (&a, &s) != 0)
    {
      puts ("2nd attr_getinheritsched failed");
      exit (1);
    }
  if (s != PTHREAD_EXPLICIT_SCHED)
    {
      printf ("inheritsched set to PTHREAD_EXPLICIT_SCHED, but got %d\n", s);
      exit (1);
    }

  e = pthread_attr_setinheritsched (&a, PTHREAD_INHERIT_SCHED);
  if (e != 0)
    {
      puts ("2nd attr_setinheritsched failed");
      exit (1);
    }
  if (pthread_attr_getinheritsched (&a, &s) != 0)
    {
      puts ("3rd attr_getinheritsched failed");
      exit (1);
    }
  if (s != PTHREAD_INHERIT_SCHED)
    {
      printf ("inheritsched set to PTHREAD_INHERIT_SCHED, but got %d\n", s);
      exit (1);
    }


  if (pthread_attr_getschedpolicy (&a, &s) != 0)
    {
      puts ("1st attr_getschedpolicy failed");
      exit (1);
    }
  /* XXX What is the correct default value.  */
  if (s != SCHED_OTHER && s != SCHED_FIFO && s != SCHED_RR)
    {
      puts ("incorrect default value for schedpolicy");
      exit (1);
    }

  e = pthread_attr_setschedpolicy (&a, SCHED_RR);
  if (e != 0)
    {
      puts ("1st attr_setschedpolicy failed");
      exit (1);
    }
  if (pthread_attr_getschedpolicy (&a, &s) != 0)
    {
      puts ("2nd attr_getschedpolicy failed");
      exit (1);
    }
  if (s != SCHED_RR)
    {
      printf ("schedpolicy set to SCHED_RR, but got %d\n", s);
      exit (1);
    }

  e = pthread_attr_setschedpolicy (&a, SCHED_FIFO);
  if (e != 0)
    {
      puts ("2nd attr_setschedpolicy failed");
      exit (1);
    }
  if (pthread_attr_getschedpolicy (&a, &s) != 0)
    {
      puts ("3rd attr_getschedpolicy failed");
      exit (1);
    }
  if (s != SCHED_FIFO)
    {
      printf ("schedpolicy set to SCHED_FIFO, but got %d\n", s);
      exit (1);
    }

  e = pthread_attr_setschedpolicy (&a, SCHED_OTHER);
  if (e != 0)
    {
      puts ("3rd attr_setschedpolicy failed");
      exit (1);
    }
  if (pthread_attr_getschedpolicy (&a, &s) != 0)
    {
      puts ("4th attr_getschedpolicy failed");
      exit (1);
    }
  if (s != SCHED_OTHER)
    {
      printf ("schedpolicy set to SCHED_OTHER, but got %d\n", s);
      exit (1);
    }


  if (pthread_attr_getscope (&a, &s) != 0)
    {
      puts ("1st attr_getscope failed");
      exit (1);
    }
  /* XXX What is the correct default value.  */
  if (s != PTHREAD_SCOPE_SYSTEM && s != PTHREAD_SCOPE_PROCESS)
    {
      puts ("incorrect default value for contentionscope");
      exit (1);
    }

  e = pthread_attr_setscope (&a, PTHREAD_SCOPE_PROCESS);
  if (e != ENOTSUP)
    {
      if (e != 0)
	{
	  puts ("1st attr_setscope failed");
	  exit (1);
	}
      if (pthread_attr_getscope (&a, &s) != 0)
	{
	  puts ("2nd attr_getscope failed");
	  exit (1);
	}
      if (s != PTHREAD_SCOPE_PROCESS)
	{
	  printf ("\
contentionscope set to PTHREAD_SCOPE_PROCESS, but got %d\n", s);
	  exit (1);
	}
    }

  e = pthread_attr_setscope (&a, PTHREAD_SCOPE_SYSTEM);
  if (e != 0)
    {
      puts ("2nd attr_setscope failed");
      exit (1);
    }
  if (pthread_attr_getscope (&a, &s) != 0)
    {
      puts ("3rd attr_getscope failed");
      exit (1);
    }
  if (s != PTHREAD_SCOPE_SYSTEM)
    {
      printf ("contentionscope set to PTHREAD_SCOPE_SYSTEM, but got %d\n", s);
      exit (1);
    }

  char buf[1];
  e = pthread_attr_setstack (&a, buf, 1);
  if (e != EINVAL)
    {
      puts ("setstack with size 1 did not produce EINVAL");
      exit (1);
    }

  e = pthread_attr_setstacksize (&a, 1);
  if (e != EINVAL)
    {
      puts ("setstacksize with size 1 did not produce EINVAL");
      exit (1);
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
