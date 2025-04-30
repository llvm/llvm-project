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
#include <stdarg.h>
#include <sysdep.h>
#include <ulimit.h>
#include <unistd.h>
#include <limits.h>
#include <sys/resource.h>

/* Function depends on CMD:
   1 = Return the limit on the size of a file, in units of 512 bytes.
   2 = Set the limit on the size of a file to NEWLIMIT.  Only the
       super-user can increase the limit.
   3 = illegal due to shared libraries; normally is
       (Return the maximum possible address of the data segment.)
   4 = Return the maximum number of files that the calling process
       can open.
   Returns -1 on errors.  */
long int
__ulimit (int cmd, ...)
{
  struct rlimit limit;
  va_list va;
  long int result = -1;

  va_start (va, cmd);

  switch (cmd)
    {
    case UL_GETFSIZE:
      /* Get limit on file size.  */
      if (__getrlimit (RLIMIT_FSIZE, &limit) == 0)
	/* Convert from bytes to 512 byte units.  */
	result =  (limit.rlim_cur == RLIM_INFINITY
		   ? LONG_MAX : limit.rlim_cur / 512);
      break;

    case UL_SETFSIZE:
      /* Set limit on file size.  */
      {
	long int newlimit = va_arg (va, long int);
	long int newlen;

	if ((rlim_t) newlimit > RLIM_INFINITY / 512)
	  {
	    limit.rlim_cur = RLIM_INFINITY;
	    limit.rlim_max = RLIM_INFINITY;
	    newlen = LONG_MAX;
	  }
	else
	  {
	    limit.rlim_cur = newlimit * 512;
	    limit.rlim_max = newlimit * 512;
	    newlen = newlimit;
	  }

	result = __setrlimit (RLIMIT_FSIZE, &limit);
	if (result != -1)
	  result = newlen;
      }
      break;

    case __UL_GETOPENMAX:
      result = __sysconf (_SC_OPEN_MAX);
      break;

    default:
      __set_errno (EINVAL);
    }

  va_end (va);

  return result;
}

weak_alias (__ulimit, ulimit);
