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

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>


int
do_test (void)
{
#if defined _POSIX_THREAD_CPUTIME && _POSIX_THREAD_CPUTIME >= 0
  clockid_t cl;
  /* This is really only a linking-test here.  */
  int e = pthread_getcpuclockid (pthread_self (), &cl);
  if (e != 0)
    {
# if _POSIX_THREAD_CPUTIME == 0
      if (sysconf (_SC_THREAD_CPUTIME) >= 0)
# endif
	{
	  puts ("cpuclock advertized, but cannot get ID");
	  exit (1);
	}
    }
#endif

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
