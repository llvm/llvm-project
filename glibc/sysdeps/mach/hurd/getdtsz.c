/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/resource.h>

/* Return the maximum number of file descriptors the current process
   could possibly have (until it raises the resource limit).  */
int
__getdtablesize (void)
{
  rlim_t limit;

  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_rlimit_lock);
  limit = _hurd_rlimits[RLIMIT_NOFILE].rlim_cur;
  __mutex_unlock (&_hurd_rlimit_lock);
  HURD_CRITICAL_END;

  /* RLIM_INFINITY is not meaningful to our caller.  -1 is a good choice
     because `sysconf (_SC_OPEN_MAX)' calls us, and -1 from sysconf means
     "no determinable limit".  */
  return limit == RLIM_INFINITY ? -1 : (int) limit;
}

weak_alias (__getdtablesize, getdtablesize)
