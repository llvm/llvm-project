/* Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2008.

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
#include <time.h>
#include <unistd.h>


#if defined _POSIX_CLOCK_SELECTION && _POSIX_CLOCK_SELECTION >= 0
static int
check (pthread_condattr_t *condattr, int pshared, clockid_t cl)
{
  clockid_t cl2;
  if (pthread_condattr_getclock (condattr, &cl2) != 0)
    {
      puts ("condattr_getclock failed");
      return 1;
    }
  if (cl != cl2)
    {
      printf ("condattr_getclock returned wrong value: %d, expected %d\n",
	      (int) cl2, (int) cl);
      return 1;
    }

  int p;
  if (pthread_condattr_getpshared (condattr, &p) != 0)
    {
      puts ("condattr_getpshared failed");
      return 1;
    }
  else if (p != pshared)
    {
      printf ("condattr_getpshared returned wrong value: %d, expected %d\n",
	      p, pshared);
      return 1;
    }

  return 0;
}

static int
run_test (clockid_t cl)
{
  pthread_condattr_t condattr;

  printf ("clock = %d\n", (int) cl);

  if (pthread_condattr_init (&condattr) != 0)
    {
      puts ("condattr_init failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_PRIVATE, CLOCK_REALTIME))
    return 1;

  if (pthread_condattr_setpshared (&condattr, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("1st condattr_setpshared failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_SHARED, CLOCK_REALTIME))
    return 1;

  if (pthread_condattr_setclock (&condattr, cl) != 0)
    {
      puts ("1st condattr_setclock failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_SHARED, cl))
    return 1;

  if (pthread_condattr_setpshared (&condattr, PTHREAD_PROCESS_PRIVATE) != 0)
    {
      puts ("2nd condattr_setpshared failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_PRIVATE, cl))
    return 1;

  if (pthread_condattr_setclock (&condattr, CLOCK_REALTIME) != 0)
    {
      puts ("2nd condattr_setclock failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_PRIVATE, CLOCK_REALTIME))
    return 1;

  if (pthread_condattr_setclock (&condattr, cl) != 0)
    {
      puts ("3rd condattr_setclock failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_PRIVATE, cl))
    return 1;

  if (pthread_condattr_setpshared (&condattr, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("3rd condattr_setpshared failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_SHARED, cl))
    return 1;

  if (pthread_condattr_setclock (&condattr, CLOCK_REALTIME) != 0)
    {
      puts ("4th condattr_setclock failed");
      return 1;
    }

  if (check (&condattr, PTHREAD_PROCESS_SHARED, CLOCK_REALTIME))
    return 1;

  if (pthread_condattr_destroy (&condattr) != 0)
    {
      puts ("condattr_destroy failed");
      return 1;
    }

  return 0;
}
#endif


static int
do_test (void)
{
#if !defined _POSIX_CLOCK_SELECTION || _POSIX_CLOCK_SELECTION == -1

  puts ("_POSIX_CLOCK_SELECTION not supported, test skipped");
  return 0;

#else

  int res = run_test (CLOCK_REALTIME);

# if defined _POSIX_MONOTONIC_CLOCK && _POSIX_MONOTONIC_CLOCK >= 0
#  if _POSIX_MONOTONIC_CLOCK == 0
  int e = sysconf (_SC_MONOTONIC_CLOCK);
  if (e < 0)
    puts ("CLOCK_MONOTONIC not supported");
  else if (e == 0)
    {
      puts ("sysconf (_SC_MONOTONIC_CLOCK) must not return 0");
      res = 1;
    }
  else
#  endif
    res |= run_test (CLOCK_MONOTONIC);
# else
  puts ("_POSIX_MONOTONIC_CLOCK not defined");
# endif

  return res;
#endif
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
