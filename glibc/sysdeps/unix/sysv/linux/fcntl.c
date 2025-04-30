/* Linux fcntl syscall implementation.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef __OFF_T_MATCHES_OFF64_T

# ifndef FCNTL_ADJUST_CMD
#  define FCNTL_ADJUST_CMD(__cmd) __cmd
# endif

int
__libc_fcntl (int fd, int cmd, ...)
{
  va_list ap;
  void *arg;

  va_start (ap, cmd);
  arg = va_arg (ap, void *);
  va_end (ap);

  cmd = FCNTL_ADJUST_CMD (cmd);

  switch (cmd)
    {
      case F_SETLKW:
      case F_SETLKW64:
	return SYSCALL_CANCEL (fcntl64, fd, cmd, arg);
      case F_OFD_SETLKW:
	{
	  struct flock *flk = (struct flock *) arg;
	  struct flock64 flk64 =
	  {
	    .l_type = flk->l_type,
	    .l_whence = flk->l_whence,
	    .l_start = flk->l_start,
	    .l_len = flk->l_len,
	    .l_pid = flk->l_pid
	  };
	  return SYSCALL_CANCEL (fcntl64, fd, cmd, &flk64);
	}
      case F_OFD_GETLK:
      case F_OFD_SETLK:
	{
	  struct flock *flk = (struct flock *) arg;
	  struct flock64 flk64 =
	  {
	    .l_type = flk->l_type,
	    .l_whence = flk->l_whence,
	    .l_start = flk->l_start,
	    .l_len = flk->l_len,
	    .l_pid = flk->l_pid
	  };
	  int ret = INLINE_SYSCALL_CALL (fcntl64, fd, cmd, &flk64);
	  if (ret == -1)
	    return -1;
	  if ((off_t) flk64.l_start != flk64.l_start
	      || (off_t) flk64.l_len != flk64.l_len)
	    {
	      __set_errno (EOVERFLOW);
	      return -1;
	    }
	  flk->l_type = flk64.l_type;
	  flk->l_whence = flk64.l_whence;
	  flk->l_start = flk64.l_start;
	  flk->l_len = flk64.l_len;
	  flk->l_pid = flk64.l_pid;
	  return ret;
	}
      /* Since only F_SETLKW{64}/F_OLD_SETLK are cancellation entrypoints and
	 only OFD locks require LFS handling, all others flags are handled
	 unmodified by calling __NR_fcntl64.  */
      default:
        return __fcntl64_nocancel_adjusted (fd, cmd, arg);
    }
}
libc_hidden_def (__libc_fcntl)

weak_alias (__libc_fcntl, __fcntl)
libc_hidden_weak (__fcntl)

# include <shlib-compat.h>
# if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_28)
int
__old_libc_fcntl64 (int fd, int cmd, ...)
{
  va_list ap;
  void *arg;

  va_start (ap, cmd);
  arg = va_arg (ap, void *);
  va_end (ap);

  /* Previous versions called __NR_fcntl64 for fcntl (which did not handle
     OFD locks in LFS mode).  */
  return __libc_fcntl64 (fd, cmd, arg);
}
compat_symbol (libc, __old_libc_fcntl64, fcntl, GLIBC_2_0);
versioned_symbol (libc, __libc_fcntl, fcntl, GLIBC_2_28);
# else
weak_alias (__libc_fcntl, fcntl)
# endif

#endif /* __OFF_T_MATCHES_OFF64_T  */
