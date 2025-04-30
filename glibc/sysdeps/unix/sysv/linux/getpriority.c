/* getpriority for Linux.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <sys/resource.h>

#include <sysdep.h>
#include <sys/syscall.h>

/* The return value of getpriority syscall is biased by this value
   to avoid returning negative values.  */
#define PZERO 20

/* Return the highest priority of any process specified by WHICH and WHO
   (see above); if WHO is zero, the current process, process group, or user
   (as specified by WHO) is used.  A lower priority number means higher
   priority.  Priorities range from PRIO_MIN to PRIO_MAX.  */

int
__getpriority (enum __priority_which which, id_t who)
{
  int res;

  res = INLINE_SYSCALL (getpriority, 2, (int) which, who);
  if (res >= 0)
    res = PZERO - res;
  return res;
}
libc_hidden_def (__getpriority)
weak_alias (__getpriority, getpriority)
