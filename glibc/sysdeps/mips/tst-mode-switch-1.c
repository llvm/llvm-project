/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <sys/prctl.h>

#if __mips_fpr != 0 || _MIPS_SPFPSET != 16
# error This test requires -mfpxx -mno-odd-spreg
#endif

/* This test verifies that mode changes do not clobber register state
   in other threads.  */

static volatile int finished;
static int mode[6] =
  {
    0,
    PR_FP_MODE_FR,
    PR_FP_MODE_FR | PR_FP_MODE_FRE,
    PR_FP_MODE_FR,
    0,
    PR_FP_MODE_FR | PR_FP_MODE_FRE
  };

static void *
thread_function (void * arg __attribute__ ((unused)))
{
  volatile int i = 0;
  volatile float f = 0.0;
  volatile double d = 0.0;

  while (!finished)
    {
      if ((float) i != f || (double) i != d)
	{
	  printf ("unexpected value: i(%d) f(%f) d(%f)\n", i, f, d);
	  exit (1);
	}

      if (i == 100)
	{
	  i = 0;
	  f = 0.0;
	  d = 0.0;
	}

      i++;
      f++;
      d++;
    }
  return NULL;
}

static int
do_test (void)
{
  int count = sysconf (_SC_NPROCESSORS_ONLN);
  if (count <= 0)
    count = 1;
  count *= 4;

  pthread_t th[count];
  int i;
  int result = 0;

  for (i = 0; i < count; ++i)
    if (pthread_create (&th[i], NULL, thread_function, 0) != 0)
      {
	printf ("creation of thread %d failed\n", i);
	exit (1);
      }

  for (i = 0 ; i < 1000000 ; i++)
    {
      if (prctl (PR_SET_FP_MODE, mode[i % 6]) != 0
	  && errno != ENOTSUP)
	{
	  printf ("prctl PR_SET_FP_MODE failed: %m\n");
	  exit (1);
	}
    }

  finished = 1;

  for (i = 0; i < count; ++i)
    {
      void *v;
      if (pthread_join (th[i], &v) != 0)
	{
	  printf ("join of thread %d failed\n", i);
	  result = 1;
	}
      else if (v != NULL)
	{
	  printf ("join %d successful, but child failed\n", i);
	  result = 1;
	}
      else
	printf ("join %d successful\n", i);
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../../test-skeleton.c"
