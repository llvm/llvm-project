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
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthreaddef.h>


extern pthread_barrier_t b;

#define TOTAL_SIGS 1000
extern int nsigs;

extern sem_t s;


static __thread void (*fp) (void);


#define THE_SIG SIGUSR1
void
handler (int sig)
{
  if (sig != THE_SIG)
    {
      write (STDOUT_FILENO, "wrong signal\n", 13);
      _exit (1);
    }

  fp ();

  if (sem_post (&s) != 0)
    {
      write (STDOUT_FILENO, "sem_post failed\n", 16);
      _exit (1);
    }
}


void *
tf (void *arg)
{
  if ((uintptr_t) pthread_self () & (TCB_ALIGNMENT - 1))
    {
      puts ("thread's struct pthread not aligned enough");
      exit (1);
    }

  if (fp != NULL)
    {
      puts ("fp not initially NULL");
      exit (1);
    }

  fp = arg;

  pthread_barrier_wait (&b);

  pthread_barrier_wait (&b);

  if (nsigs != TOTAL_SIGS)
    {
      puts ("barrier_wait prematurely returns");
      exit (1);
    }

  return NULL;
}
