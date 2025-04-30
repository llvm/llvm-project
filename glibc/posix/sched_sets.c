/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <sched.h>
#include <sys/types.h>


/* Set scheduling algorithm and/or parameters for a process.  */
int
__sched_setscheduler (pid_t pid, int policy, const struct sched_param *param)
{
  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (__sched_setscheduler)
stub_warning (sched_setscheduler)

weak_alias (__sched_setscheduler, sched_setscheduler)
