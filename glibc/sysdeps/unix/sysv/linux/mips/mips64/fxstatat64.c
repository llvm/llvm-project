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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/stat.h>
#include <kernel_stat.h>
#include <sysdep.h>
#include <xstatconv.h>

/* Get information about the file NAME in BUF.  */

int
__fxstatat64 (int vers, int fd, const char *file, struct stat64 *st, int flag)
{
  if (vers == _STAT_VER_LINUX)
    {
      struct kernel_stat kst;
      int r = INLINE_SYSCALL_CALL (newfstatat, fd, file, &kst, flag);;
      return r ?: __xstat64_conv (vers, &kst, st);
    }
  return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
}
