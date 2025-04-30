/* Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <sys/uio.h>
#include <sysdep-cancel.h>

#ifdef __ASSUME_PWRITEV

ssize_t
pwritev64 (int fd, const struct iovec *vector, int count, off64_t offset)
{
  return SYSCALL_CANCEL (pwritev, fd, vector, count, LO_HI_LONG (offset));
}
#else
static ssize_t __atomic_pwritev64_replacement (int, const struct iovec *,
					       int, off64_t);
ssize_t
pwritev64 (int fd, const struct iovec *vector, int count, off64_t offset)
{
  ssize_t result = SYSCALL_CANCEL (pwritev, fd, vector, count,
				   LO_HI_LONG (offset));
  if (result >= 0 || errno != ENOSYS)
    return result;
  return __atomic_pwritev64_replacement (fd, vector, count, offset);
}
# define PWRITEV static __atomic_pwritev64_replacement
# define PWRITE __pwrite64
# define OFF_T off64_t
# include <sysdeps/posix/pwritev_common.c>
#endif
libc_hidden_def (pwritev64)

#ifdef __OFF_T_MATCHES_OFF64_T
strong_alias (pwritev64, pwritev)
libc_hidden_def (pwritev)
#endif
