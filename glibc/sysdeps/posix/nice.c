/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <unistd.h>
#include <sys/resource.h>

/* Increment the scheduling priority of the calling process by INCR.
   The superuser may use a negative INCR to decrement the priority.  */
int
nice (int incr)
{
  int save;
  int prio;
  int result;

  /* -1 is a valid priority, so we use errno to check for an error.  */
  save = errno;
  __set_errno (0);
  prio = __getpriority (PRIO_PROCESS, 0);
  if (prio == -1)
    {
      if (errno != 0)
	return -1;
    }

  result = __setpriority (PRIO_PROCESS, 0, prio + incr);
  if (result == -1)
    {
      if (errno == EACCES)
	__set_errno (EPERM);
      return -1;
    }

  __set_errno (save);
  return __getpriority (PRIO_PROCESS, 0);
}
