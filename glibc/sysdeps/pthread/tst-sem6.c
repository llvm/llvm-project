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
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>


static void
handler (int sig)
{
  struct sigaction sa;

  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  sigemptyset (&sa.sa_mask);

  sigaction (SIGALRM, &sa, NULL);

  /* Rearm the timer.  */
  alarm (1);
}


static int
do_test (void)
{
  sem_t s;
  struct sigaction sa;

  sa.sa_handler = handler;
  sa.sa_flags = 0;
  sigemptyset (&sa.sa_mask);

  sigaction (SIGALRM, &sa, NULL);

  if (sem_init (&s, 0, 0) == -1)
    {
      puts ("init failed");
      return 1;
    }

  /* Set an alarm for 1 second.  The wrapper will expect this.  */
  alarm (1);

  int res = sem_wait (&s);
  if (res == 0)
    {
      puts ("wait succeeded");
      return 1;
    }
  if (res != -1 || errno != EINTR)
    {
      puts ("wait didn't fail with EINTR");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
