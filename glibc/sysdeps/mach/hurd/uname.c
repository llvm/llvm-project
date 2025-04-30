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

#include <unistd.h>
#include <sys/utsname.h>
#include <hurd.h>
#include <hurd/startup.h>

int
__uname (struct utsname *uname)
{
  error_t err;

  if (err = __USEPORT (PROC, __proc_uname (port, uname)))
    return __hurd_fail (err);

  /* Fill in the hostname, which the proc server doesn't know.  */
  err = errno;
  if (__gethostname (uname->nodename, sizeof uname->nodename) < 0)
    {
      if (errno == ENAMETOOLONG)
	/* Ignore the error of the buffer being too small.
	   It is of fixed size, nothing to do about it.  */
	errno = err;
      else
	return -1;
    }

  return 0;
}
weak_alias (__uname, uname)
libc_hidden_def (__uname)
libc_hidden_def (uname)
