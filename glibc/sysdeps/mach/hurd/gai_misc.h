/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <signal.h>
#include <pthread.h>

#define gai_start_notify_thread __gai_start_notify_thread
#define gai_create_helper_thread __gai_create_helper_thread

extern inline void
__gai_start_notify_thread (void)
{
  sigset_t ss;
  sigemptyset (&ss);
  int sigerr __attribute__ ((unused));
  sigerr = pthread_sigmask (SIG_SETMASK, &ss, NULL);
  assert_perror (sigerr);
}

extern inline int
__gai_create_helper_thread (pthread_t *threadp, void *(*tf) (void *),
			    void *arg)
{
  pthread_attr_t attr;

  /* Make sure the thread is created detached.  */
  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

  /* The helper thread needs only very little resources.  */
  (void) pthread_attr_setstacksize (&attr, 0x10000);

  /* Block all signals in the helper thread.  To do this thoroughly we
     temporarily have to block all signals here.  */
  sigset_t ss;
  sigset_t oss;
  sigfillset (&ss);
  int sigerr __attribute__ ((unused));
  sigerr = pthread_sigmask (SIG_SETMASK, &ss, &oss);
  assert_perror (sigerr);

  int ret = pthread_create (threadp, &attr, tf, arg);

  /* Restore the signal mask.  */
  sigerr = pthread_sigmask (SIG_SETMASK, &oss, NULL);
  assert_perror (sigerr);

  (void) pthread_attr_destroy (&attr);
  return ret;
}

#include_next <gai_misc.h>
