/* Open a file by name.  Stub version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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


/* Open FILE with access OFLAG.  If O_CREAT or O_TMPFILE is in OFLAG,
   a third argument is the file protection.  */
int
__libc_open (const char *file, int oflag)
{
  int mode;

  if (file == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  if (__OPEN_NEEDS_MODE (oflag))
    {
      va_list arg;
      va_start(arg, oflag);
      mode = va_arg(arg, int);
      va_end(arg);
    }

  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (__libc_open)
weak_alias (__libc_open, __open)
libc_hidden_weak (__open)
weak_alias (__libc_open, open)

stub_warning (open)

/* __open_2 is a generic wrapper that calls __open.
   So give a stub warning for that symbol too.  */
stub_warning (__open_2)
