/* f_setlk -- locking part of fcntl
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/* XXX
   We need new RPCs to support POSIX.1 fcntl file locking!!
   For the time being we support the whole-file case only,
   with all kinds of WRONG WRONG WRONG semantics,
   by using flock.  This is definitely the Wrong Thing,
   but it might be better than nothing (?).  */
int
__f_setlk (int fd, int type, int whence, __off64_t start, __off64_t len, int wait)
{
  int cmd = 0;

  switch (type)
    {
    case F_RDLCK: cmd = LOCK_SH; break;
    case F_WRLCK: cmd = LOCK_EX; break;
    case F_UNLCK: cmd = LOCK_UN; break;
    default:
      errno = EINVAL;
      return -1;
    }

  if (cmd != LOCK_UN && wait == 0)
    cmd |= LOCK_NB;

  if (whence == SEEK_CUR)
    {
      /* In case the target position is 0, we can support it below.  */
      __off64_t cur = __lseek64 (fd, 0, SEEK_CUR);

      if (cur >= 0)
	{
	  start = cur + start;
	  whence = SEEK_SET;
	}
    }

  switch (whence)
    {
    case SEEK_SET:
      if (start == 0 && len == 0) /* Whole file request.  */
	break;
      /* It seems to be common for applications to lock the first
	 byte of the file when they are really doing whole-file locking.
	 So, since it's so wrong already, might as well do that too.  */
      if (start == 0 && len == 1)
	break;
      /* FALLTHROUGH */
    case SEEK_CUR:
    case SEEK_END:
      errno = ENOTSUP;
      return -1;
    default:
      errno = EINVAL;
      return -1;
    }

  return __flock (fd, cmd);
}
