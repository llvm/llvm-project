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

#include <unistd.h>
#include <hurd.h>
#include <hurd/port.h>
#include <sysdep.h>
#include <sys/wait.h>

void
_hurd_exit (int status)
{
  /* Give the proc server our exit status.  */
  __USEPORT (PROC, __proc_mark_exit (port, status, 0));

  /* Commit suicide.  */
  __task_terminate (__mach_task_self ());

  /* Perhaps the cached mach_task_self was bogus.  */
  __task_terminate ((__mach_task_self) ());

  /* This sucker really doesn't want to die.  */
  while (1)
    {
#ifdef LOSE
      LOSE;
#else
      volatile const int zero = 0, one = 1;
      volatile int lossage = one / zero;
#endif
    }
}

void
_exit (int status)
{
  _hurd_exit (W_EXITCODE (status, 0));
}
libc_hidden_def (_exit)
#ifndef NO_RTLD_HIDDEN
rtld_hidden_def (_exit)
#endif
weak_alias (_exit, _Exit)
