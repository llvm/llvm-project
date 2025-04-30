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
#include <signal.h>
#include <unistd.h>
#include <support/xthread.h>
#include <support/xsignal.h>
#include <support/xthread.h>


static int the_sig;


static void
eintr_handler (int sig)
{
  if (sig != the_sig)
    {
      write (STDOUT_FILENO, "eintr_handler: signal number wrong\n", 35);
      _exit (1);
    }
  write (STDOUT_FILENO, ".", 1);
}


static void *
eintr_source (void *arg)
{
  struct timespec ts = { .tv_sec = 0, .tv_nsec = 500000 };

  if (arg == NULL)
    {
      sigset_t ss;
      sigemptyset (&ss);
      sigaddset (&ss, the_sig);
      xpthread_sigmask (SIG_BLOCK, &ss, NULL);
    }

  while (1)
    {
      if (arg != NULL)
	pthread_kill (*(pthread_t *) arg, the_sig);
      else
	kill (getpid (), the_sig);

      nanosleep (&ts, NULL);
    }

  /* NOTREACHED */
  return NULL;
}


static void
setup_eintr (int sig, pthread_t *thp)
{
  struct sigaction sa;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;
  sa.sa_handler = eintr_handler;
  if (sigaction (sig, &sa, NULL) != 0)
    {
      puts ("setup_eintr: sigaction failed");
      exit (1);
    }
  the_sig = sig;

  /* Create the thread which will fire off the signals.  */
  xpthread_create (NULL, eintr_source, thp);
}
