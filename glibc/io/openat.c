/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/stat.h>
#include <kernel-features.h>


/* Open FILE with access OFLAG.  Interpret relative paths relative to
   the directory associated with FD.  If O_CREAT or O_TMPFILE is in OFLAG, a
   third argument is the file protection.  */
int
__openat (int fd, const char *file, int oflag, ...)
{
  int mode;

  if (file == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  if (fd != AT_FDCWD && file[0] != '/')
    {
      /* Check FD is associated with a directory.  */
      struct stat64 st;
      if (__fstat64 (fd, &st) != 0)
	return -1;

      if (!S_ISDIR (st.st_mode))
	{
	  __set_errno (ENOTDIR);
	  return -1;
	}
    }

  if (__OPEN_NEEDS_MODE (oflag))
    {
      va_list arg;
      va_start (arg, oflag);
      mode = va_arg (arg, int);
      va_end (arg);

      ignore_value (mode);
    }

  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (__openat)
weak_alias (__openat, openat)
stub_warning (openat)

/* __openat_2 is a generic wrapper that calls __openat.
   So give a stub warning for that symbol too.  */
stub_warning (__openat_2)
