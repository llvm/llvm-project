/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2006.

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
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

/* This test is Linux specific.  */
#define CHECK_TPP_PRIORITY(normal, boosted) \
  do								\
    {								\
      pid_t tid = syscall (__NR_gettid);			\
								\
      struct sched_param cep_sp;				\
      int cep_policy;						\
      if (pthread_getschedparam (pthread_self (), &cep_policy,	\
				 &cep_sp) != 0)			\
	{							\
	  puts ("getschedparam failed");			\
	  ret = 1;						\
	}							\
      else if (cep_sp.sched_priority != (normal))		\
	{							\
	  printf ("unexpected priority %d != %d\n",		\
		  cep_sp.sched_priority, (normal));		\
	}							\
      if (syscall (__NR_sched_getparam, tid, &cep_sp) == 0	\
	  && cep_sp.sched_priority != (boosted))		\
	{							\
	  printf ("unexpected boosted priority %d != %d\n",	\
		  cep_sp.sched_priority, (boosted));		\
	  ret = 1;						\
	}							\
    }								\
  while (0)

int fifo_min, fifo_max;

void
init_tpp_test (void)
{
  fifo_min = sched_get_priority_min (SCHED_FIFO);
  if (fifo_min < 0)
    {
      printf ("couldn't get min priority for SCHED_FIFO: %m\n");
      exit (1);
    }

  fifo_max = sched_get_priority_max (SCHED_FIFO);
  if (fifo_max < 0)
    {
      printf ("couldn't get max priority for SCHED_FIFO: %m\n");
      exit (1);
    }

  if (fifo_min > 4 || fifo_max < 10)
    {
      printf ("%d..%d SCHED_FIFO priority interval not suitable for this test\n",
	      fifo_min, fifo_max);
      exit (0);
    }

  struct sched_param sp;
  memset (&sp, 0, sizeof (sp));
  sp.sched_priority = 4;
  int e = pthread_setschedparam (pthread_self (), SCHED_FIFO, &sp);
  if (e != 0)
    {
      errno = e;
      printf ("cannot set scheduling params: %m\n");
      exit (0);
    }
}
