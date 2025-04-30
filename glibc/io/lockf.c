/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <bits/types.h>

#ifndef __OFF_T_MATCHES_OFF64_T

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

/* lockf is a simplified interface to fcntl's locking facilities.  */
int
lockf (int fd, int cmd, off_t len)
{
  /* lockf is always relative to the current file position.  */
  struct flock fl = {
    .l_type = F_WRLCK,
    .l_whence = SEEK_CUR,
    .l_len = len
  };

  /* lockf() is a cancellation point but so is fcntl() if F_SETLKW is
     used.  Therefore we don't have to care about cancellation here,
     the fcntl() function will take care of it.  */
  switch (cmd)
    {
    case F_TEST:
      /* Test the lock: return 0 if FD is unlocked or locked by this process;
	 return -1, set errno to EACCES, if another process holds the lock.  */
      fl.l_type = F_RDLCK;
      if (__fcntl (fd, F_GETLK, &fl) < 0)
	return -1;
      if (fl.l_type == F_UNLCK || fl.l_pid == __getpid ())
	return 0;
      __set_errno (EACCES);
      return -1;
    case F_ULOCK:
      fl.l_type = F_UNLCK;
      return __fcntl (fd, F_SETLK, &fl);
    case F_LOCK:
      return __fcntl (fd, F_SETLKW, &fl);
    case F_TLOCK:
      return __fcntl (fd, F_SETLK, &fl);
    }
  __set_errno (EINVAL);
  return -1;
}
#endif
