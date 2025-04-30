/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2009.

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
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <pthreadP.h>
#include <tls.h>
#include <sysdep.h>
#include <shlib-compat.h>

int
__pthread_sigqueue (pthread_t threadid, int signo, const union sigval value)
{
#ifdef __NR_rt_tgsigqueueinfo
  struct pthread *pd = (struct pthread *) threadid;

  /* Force load of pd->tid into local variable or register.  Otherwise
     if a thread exits between ESRCH test and tgkill, we might return
     EINVAL, because pd->tid would be cleared by the kernel.  */
  pid_t tid = atomic_forced_read (pd->tid);
  if (__glibc_unlikely (tid <= 0))
    /* Not a valid thread handle.  */
    return ESRCH;

  /* Disallow sending the signal we use for cancellation, timers,
     for the setxid implementation.  */
  if (signo == SIGCANCEL || signo == SIGTIMER || signo == SIGSETXID)
    return EINVAL;

  pid_t pid = getpid ();

  /* Set up the siginfo_t structure.  */
  siginfo_t info;
  memset (&info, '\0', sizeof (siginfo_t));
  info.si_signo = signo;
  info.si_code = SI_QUEUE;
  info.si_pid = pid;
  info.si_uid = __getuid ();
  info.si_value = value;

  /* We have a special syscall to do the work.  */
  int val = INTERNAL_SYSCALL_CALL (rt_tgsigqueueinfo, pid, tid, signo,
				   &info);
  return (INTERNAL_SYSCALL_ERROR_P (val)
	  ? INTERNAL_SYSCALL_ERRNO (val) : 0);
#else
  return ENOSYS;
#endif
}
versioned_symbol (libc, __pthread_sigqueue, pthread_sigqueue, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_11, GLIBC_2_34)
compat_symbol (libpthread, __pthread_sigqueue, pthread_sigqueue, GLIBC_2_11);
#endif
