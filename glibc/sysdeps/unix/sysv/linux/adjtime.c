/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#include <limits.h>
#include <sys/time.h>
#include <sys/timex.h>
#include <sysdep.h>

#define MAX_SEC	(INT_MAX / 1000000L - 2)
#define MIN_SEC	(INT_MIN / 1000000L + 2)

int
__adjtime64 (const struct __timeval64 *itv, struct __timeval64 *otv)
{
  struct __timex64 tntx;

  if (itv)
    {
      struct __timeval64 tmp;

      /* We will do some check here. */
      tmp.tv_sec = itv->tv_sec + itv->tv_usec / 1000000L;
      tmp.tv_usec = itv->tv_usec % 1000000L;
      if (tmp.tv_sec > MAX_SEC || tmp.tv_sec < MIN_SEC)
	return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
      tntx.offset = tmp.tv_usec + tmp.tv_sec * 1000000L;
      tntx.modes = ADJ_OFFSET_SINGLESHOT;
    }
  else
    tntx.modes = ADJ_OFFSET_SS_READ;

  if (__glibc_unlikely (__clock_adjtime64 (CLOCK_REALTIME, &tntx) < 0))
    return -1;

  if (otv)
    {
      if (tntx.offset < 0)
	{
	  otv->tv_usec = -(-tntx.offset % 1000000);
	  otv->tv_sec  = -(-tntx.offset / 1000000);
	}
      else
	{
	  otv->tv_usec = tntx.offset % 1000000;
	  otv->tv_sec  = tntx.offset / 1000000;
	}
    }
  return 0;
}

#if __TIMESIZE != 64
libc_hidden_def (__adjtime64)

int
__adjtime (const struct timeval *itv, struct timeval *otv)
{
  struct __timeval64 itv64, *pitv64 = NULL;
  struct __timeval64 otv64;
  int retval;

  if (itv != NULL)
    {
      itv64 = valid_timeval_to_timeval64 (*itv);
      pitv64 = &itv64;
    }
  retval = __adjtime64 (pitv64, otv != NULL ? &otv64 : NULL);
  if (otv != NULL)
    *otv = valid_timeval64_to_timeval (otv64);

  return retval;
}
#endif

#ifdef VERSION_adjtime
weak_alias (__adjtime, __wadjtime);
default_symbol_version (__wadjtime, adjtime, VERSION_adjtime);
#else
weak_alias (__adjtime, adjtime)
#endif
