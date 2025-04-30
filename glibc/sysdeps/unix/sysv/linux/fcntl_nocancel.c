/* Linux fcntl syscall implementation -- non-cancellable.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <fcntl.h>
#include <stdarg.h>
#include <errno.h>
#include <sysdep-cancel.h>
#include <not-cancel.h>

#ifndef __NR_fcntl64
# define __NR_fcntl64 __NR_fcntl
#endif

#ifndef FCNTL_ADJUST_CMD
# define FCNTL_ADJUST_CMD(__cmd) __cmd
#endif

int
__fcntl64_nocancel (int fd, int cmd, ...)
{
  va_list ap;
  void *arg;

  va_start (ap, cmd);
  arg = va_arg (ap, void *);
  va_end (ap);

  cmd = FCNTL_ADJUST_CMD (cmd);

  return __fcntl64_nocancel_adjusted (fd, cmd, arg);
}
hidden_def (__fcntl64_nocancel)

int
__fcntl64_nocancel_adjusted (int fd, int cmd, void *arg)
{
  if (cmd == F_GETOWN)
    {
      struct f_owner_ex fex;
      int res = INTERNAL_SYSCALL_CALL (fcntl64, fd, F_GETOWN_EX, &fex);
      if (!INTERNAL_SYSCALL_ERROR_P (res))
	return fex.type == F_OWNER_GID ? -fex.pid : fex.pid;

      return INLINE_SYSCALL_ERROR_RETURN_VALUE
        (INTERNAL_SYSCALL_ERRNO (res));
    }

  return INLINE_SYSCALL_CALL (fcntl64, fd, cmd, (void *) arg);
}
