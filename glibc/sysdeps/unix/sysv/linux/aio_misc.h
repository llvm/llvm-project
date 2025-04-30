/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#ifndef _AIO_MISC_H
# include_next <aio_misc.h>
# include <limits.h>
# include <pthread.h>
# include <signal.h>
# include <sysdep.h>

# define aio_start_notify_thread __aio_start_notify_thread
# define aio_create_helper_thread __aio_create_helper_thread

extern inline void
__aio_start_notify_thread (void)
{
  sigset_t ss;
  sigemptyset (&ss);
  INTERNAL_SYSCALL_CALL (rt_sigprocmask, SIG_SETMASK, &ss, NULL,
			 __NSIG_BYTES);
}

extern inline int
__aio_create_helper_thread (pthread_t *threadp, void *(*tf) (void *),
			    void *arg)
{
  pthread_attr_t attr;

  /* Make sure the thread is created detached.  */
  __pthread_attr_init (&attr);
  __pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

  /* The helper thread needs only very little resources.  */
  __pthread_attr_setstacksize (&attr, __pthread_get_minstack (&attr));

  /* Block all signals in the helper thread.  To do this thoroughly we
     temporarily have to block all signals here.  */
  sigset_t ss;
  sigset_t oss;
  sigfillset (&ss);
  INTERNAL_SYSCALL_CALL (rt_sigprocmask, SIG_SETMASK, &ss, &oss,
			 __NSIG_BYTES);

  int ret = __pthread_create (threadp, &attr, tf, arg);

  /* Restore the signal mask.  */
  INTERNAL_SYSCALL_CALL (rt_sigprocmask, SIG_SETMASK, &oss, NULL,
			 __NSIG_BYTES);

  __pthread_attr_destroy (&attr);
  return ret;
}
#endif
