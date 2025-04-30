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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <pthread.h>
#include <sys/prctl.h>

#if __mips_fpr != 0 || _MIPS_SPFPSET != 16
# error This test requires -mfpxx -mno-odd-spreg
#endif

/* This test verifies that all threads in a process see a mode
   change when any thread causes a mode change.  */

static int mode[6] =
  {
    0,
    PR_FP_MODE_FR,
    PR_FP_MODE_FR | PR_FP_MODE_FRE,
    PR_FP_MODE_FR,
    0,
    PR_FP_MODE_FR | PR_FP_MODE_FRE
  };
static volatile int current_mode;
static volatile int finished;
static pthread_barrier_t barr_ready;
static pthread_barrier_t barr_cont;

static void *
thread_function (void * arg __attribute__ ((unused)))
{
  while (!finished)
    {
      int res = pthread_barrier_wait (&barr_ready);

      if (res != 0 && res != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("barrier wait failed: %m\n");
	  exit (1);
	}

      int mode = prctl (PR_GET_FP_MODE);

      if (mode != current_mode)
	{
	  printf ("unexpected mode: %d != %d\n", mode, current_mode);
	  exit (1);
	}

      res = pthread_barrier_wait (&barr_cont);

      if (res != 0 && res != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("barrier wait failed: %m\n");
	  exit (1);
	}
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

  if (pthread_barrier_init (&barr_ready, NULL, count + 1) != 0)
    {
      printf ("failed to initialize barrier: %m\n");
      exit (1);
    }

  if (pthread_barrier_init (&barr_cont, NULL, count + 1) != 0)
    {
      printf ("failed to initialize barrier: %m\n");
      exit (1);
    }

  for (i = 0; i < count; ++i)
    if (pthread_create (&th[i], NULL, thread_function, 0) != 0)
      {
	printf ("creation of thread %d failed\n", i);
	exit (1);
      }

  for (i = 0 ; i < 7 ; i++)
    {
      if (prctl (PR_SET_FP_MODE, mode[i % 6]) != 0)
	{
	  if (errno != ENOTSUP)
	    {
	      printf ("prctl PR_SET_FP_MODE failed: %m");
	      exit (1);
	    }
	}
      else
	current_mode = mode[i % 6];


      int res = pthread_barrier_wait (&barr_ready);

      if (res != 0 && res != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("barrier wait failed: %m\n");
	  exit (1);
	}

      if (i == 6)
	finished = 1;

      res = pthread_barrier_wait (&barr_cont);

      if (res != 0 && res != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("barrier wait failed: %m\n");
	  exit (1);
	}
    }

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
