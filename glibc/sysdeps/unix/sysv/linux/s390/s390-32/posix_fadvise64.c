/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
#include <sysdep.h>

int __posix_fadvise64_l64 (int fd, off64_t offset, off64_t len, int advise);
int __posix_fadvise64_l32 (int fd, off64_t offset, size_t len, int advise);

/* Advice the system about the expected behaviour of the application with
   respect to the file associated with FD.  */

struct fadvise64_64_layout
{
  int fd;
  off64_t offset;
  off64_t len;
  int advise;
};

int
__posix_fadvise64_l64 (int fd, off64_t offset, off64_t len, int advise)
{
  struct fadvise64_64_layout parameters;

  parameters.fd = fd;
  parameters.offset = offset;
  parameters.len = len;
  parameters.advise = advise;
  int ret = INTERNAL_SYSCALL_CALL (fadvise64_64, &parameters);
  if (!INTERNAL_SYSCALL_ERROR_P (ret))
    return 0;
  return INTERNAL_SYSCALL_ERRNO (ret);
}

#include <shlib-compat.h>

#if SHLIB_COMPAT(libc, GLIBC_2_2, GLIBC_2_3_3)

int
attribute_compat_text_section
__posix_fadvise64_l32 (int fd, off64_t offset, size_t len, int advise)
{
  return __posix_fadvise64_l64 (fd, offset, len, advise);
}

versioned_symbol (libc, __posix_fadvise64_l64, posix_fadvise64, GLIBC_2_3_3);
compat_symbol (libc, __posix_fadvise64_l32, posix_fadvise64, GLIBC_2_2);
#else
strong_alias (__posix_fadvise64_l64, posix_fadvise64);
#endif
