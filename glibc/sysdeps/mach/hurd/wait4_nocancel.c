/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <hurd.h>
#include <hurd/port.h>
#include <not-cancel.h>

pid_t
__wait4_nocancel (pid_t pid, int *stat_loc, int options, struct rusage *usage)
{
  pid_t dead;
  error_t err;
  struct rusage ignored;
  int sigcode;
  int dummy;

  err = __USEPORT (PROC, __proc_wait (port, pid, options,
				      stat_loc ?: &dummy, &sigcode,
				      usage ?: &ignored, &dead));
  switch (err)
    {
    case 0:			/* Got a child.  */
      return dead;
    case EAGAIN:
      /* The RPC returns this error when the WNOHANG flag is set and no
	 selected children are dead (but some are living).  In that
	 situation, our return value is zero.  (The RPC can't return zero
	 for DEAD without also returning some garbage for the other out
	 parameters, so an error return is much more natural here.  Hence
	 the difference between the RPC and the POSIX.1 interface.  */
      return (pid_t) 0;
    default:
      return (pid_t) __hurd_fail (err);
    }
}

libc_hidden_def (__wait4_nocancel)
