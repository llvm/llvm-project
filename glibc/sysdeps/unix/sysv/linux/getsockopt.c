/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <sys/socket.h>
#include <time.h>
#include <sysdep.h>
#include <socketcall.h>
#include <socket-constants-time64.h>

static int
getsockopt_syscall (int fd, int level, int optname, void *optval,
		    socklen_t *len)
{
#ifdef __ASSUME_GETSOCKOPT_SYSCALL
  return INLINE_SYSCALL (getsockopt, 5, fd, level, optname, optval, len);
#else
  return SOCKETCALL (getsockopt, fd, level, optname, optval, len);
#endif
}

#ifndef __ASSUME_TIME64_SYSCALLS
static int
getsockopt32 (int fd, int level, int optname, void *optval,
	      socklen_t *len)
{
  int r = -1;

  if (level != SOL_SOCKET)
    return r;

  switch (optname)
    {
    case COMPAT_SO_RCVTIMEO_NEW:
    case COMPAT_SO_SNDTIMEO_NEW:
      {
	if (optname == COMPAT_SO_RCVTIMEO_NEW)
	  optname = COMPAT_SO_RCVTIMEO_OLD;
	if (optname == COMPAT_SO_SNDTIMEO_NEW)
	  optname = COMPAT_SO_SNDTIMEO_OLD;

	struct __timeval32 tv32;
	r = getsockopt_syscall (fd, level, optname, &tv32,
				(socklen_t[]) { sizeof tv32 });
	if (r < 0)
	  break;

	/* POSIX states that if the size of the option value is greater than
	   then option length, the option value argument shall be silently
	   truncated.  */
	if (*len >= sizeof (struct __timeval64))
	  {
	    struct __timeval64 *tv64 = (struct __timeval64 *) optval;
	    *tv64 = valid_timeval32_to_timeval64 (tv32);
	    *len = sizeof (*tv64);
	  }
	else
	  memcpy (optval, &tv32, sizeof tv32);
      }
      break;

    case COMPAT_SO_TIMESTAMP_NEW:
    case COMPAT_SO_TIMESTAMPNS_NEW:
      {
	if (optname == COMPAT_SO_TIMESTAMP_NEW)
	  optname = COMPAT_SO_TIMESTAMP_OLD;
	if (optname == COMPAT_SO_TIMESTAMPNS_NEW)
	  optname = COMPAT_SO_TIMESTAMPNS_OLD;
	r = getsockopt_syscall (fd, level, optname, optval, len);
      }
      break;
    }

  return r;
}
#endif

int
__getsockopt (int fd, int level, int optname, void *optval, socklen_t *len)
{
  int r = getsockopt_syscall (fd, level, optname, optval, len);

#ifndef __ASSUME_TIME64_SYSCALLS
  if (r == -1 && errno == ENOPROTOOPT)
    r = getsockopt32 (fd, level, optname, optval, len);
#endif

 return r;
}
weak_alias (__getsockopt, getsockopt)
#if __TIMESIZE != 64
weak_alias (__getsockopt, __getsockopt64)
#endif
