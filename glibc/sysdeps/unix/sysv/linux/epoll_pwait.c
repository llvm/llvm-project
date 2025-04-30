/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
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
#include <signal.h>
#include <unistd.h>
#include <sys/epoll.h>

#include <sysdep-cancel.h>
#include <sys/syscall.h>

/* Wait for events on an epoll instance "epfd". Returns the number of
   triggered events returned in "events" buffer. Or -1 in case of
   error with the "errno" variable set to the specific error code. The
   "events" parameter is a buffer that will contain triggered
   events. The "maxevents" is the maximum number of events to be
   returned ( usually size of "events" ). The "timeout" parameter
   specifies the maximum wait time in milliseconds (-1 == infinite).
   The thread's signal mask is temporarily and atomically replaced with
   the one provided as parameter.  */

int epoll_pwait (int epfd, struct epoll_event *events,
		 int maxevents, int timeout,
		 const sigset_t *set)
{
  return SYSCALL_CANCEL (epoll_pwait, epfd, events, maxevents,
			 timeout, set, __NSIG_BYTES);
}
libc_hidden_def (epoll_pwait)
