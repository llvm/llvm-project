/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

/* Return the session ID of FD.  */
pid_t
tcgetsid (int fd)
{
  pid_t pgrp;
  pid_t sid;
#ifdef TIOCGSID
  static int tiocgsid_does_not_work;

  if (! tiocgsid_does_not_work)
    {
      int serrno = errno;
      int sid;

      if (__ioctl (fd, TIOCGSID, &sid) < 0)
	{
	  if (errno == EINVAL)
	    {
	      tiocgsid_does_not_work = 1;
	      __set_errno (serrno);
	    }
	  else
	    return (pid_t) -1;
	}
      else
	return (pid_t) sid;
    }
#endif

  pgrp = tcgetpgrp (fd);
  if (pgrp == -1)
    return (pid_t) -1;

  sid = getsid (pgrp);
  if (sid == -1 && errno == ESRCH)
    __set_errno (ENOTTY);

  return sid;
}
