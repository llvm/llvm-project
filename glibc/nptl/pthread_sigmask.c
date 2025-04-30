/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <signal.h>
#include <pthreadP.h>
#include <sysdep.h>
#include <shlib-compat.h>

int
__pthread_sigmask (int how, const sigset_t *newmask, sigset_t *oldmask)
{
  sigset_t local_newmask;

  /* The only thing we have to make sure here is that SIGCANCEL and
     SIGSETXID is not blocked.  */
  if (newmask != NULL
      && (__glibc_unlikely (__sigismember (newmask, SIGCANCEL))
         || __glibc_unlikely (__sigismember (newmask, SIGSETXID))))
    {
      local_newmask = *newmask;
      __clear_internal_signals (&local_newmask);
      newmask = &local_newmask;
    }

  /* We know that realtime signals are available if NPTL is used.  */
  int result = INTERNAL_SYSCALL_CALL (rt_sigprocmask, how, newmask,
				      oldmask, __NSIG_BYTES);

  return (INTERNAL_SYSCALL_ERROR_P (result)
	  ? INTERNAL_SYSCALL_ERRNO (result)
	  : 0);
}
libc_hidden_def (__pthread_sigmask)

versioned_symbol (libc, __pthread_sigmask, pthread_sigmask, GLIBC_2_32);
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_32)
strong_alias (__pthread_sigmask, __pthread_sigmask_2);
compat_symbol (libc, __pthread_sigmask_2, pthread_sigmask, GLIBC_2_0);
#endif
