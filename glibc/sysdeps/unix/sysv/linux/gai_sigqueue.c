/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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
#include <netdb.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#include <sysdep.h>
#include <sys/syscall.h>

#include "gai_misc.h"

/* Return any pending signal or wait for one for the given time.  */
int
__gai_sigqueue (int sig, const union sigval val, pid_t caller_pid)
{
  siginfo_t info;

  /* First, clear the siginfo_t structure, so that we don't pass our
     stack content to other tasks.  */
  memset (&info, 0, sizeof (siginfo_t));
  /* We must pass the information about the data in a siginfo_t value.  */
  info.si_signo = sig;
  info.si_code = SI_ASYNCNL;
  info.si_pid = caller_pid;
  info.si_uid = __getuid ();
  info.si_value = val;

  return INLINE_SYSCALL (rt_sigqueueinfo, 3, info.si_pid, sig, &info);
}
libc_hidden_def (__gai_sigqueue)
