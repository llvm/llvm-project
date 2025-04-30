/* pause -- suspend the process until a signal arrives.  POSIX.1 version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <signal.h>
#include <unistd.h>
#include <sysdep-cancel.h>
#include <sigsetops.h>

/* Suspend the process until a signal arrives.
   This always returns -1 and sets errno to EINTR.  */

int
__libc_pause (void)
{
  sigset_t set;

  __sigemptyset (&set);
  __sigprocmask (SIG_BLOCK, NULL, &set);

  /* pause is a cancellation point, but so is sigsuspend.
     So no need for anything special here.  */

  return __sigsuspend (&set);
}
weak_alias (__libc_pause, pause)
