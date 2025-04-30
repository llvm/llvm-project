/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <sysdep.h>
#include <sys/syscall.h>

int
__gethostname (char *name, size_t len)
{
  int result;

  result = INLINE_SYSCALL (gethostname, 2, name, len);

  if (result == 0
      /* See whether the string is terminated.  If not we will return
	 an error.  */
      && memchr (name, '\0', len) == NULL)
    {
      __set_errno (EOVERFLOW);
      result = -1;
    }

  return result;
}

weak_alias (__gethostname, gethostname)
