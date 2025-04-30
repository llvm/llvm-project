/* fxstatat used on fstatat, Linux implementation.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <sys/stat.h>
#include <fcntl.h>
#include <kernel_stat.h>
#include <sysdep.h>

#if !XSTAT_IS_XSTAT64
# include <xstatconv.h>
# include <xstatover.h>
# include <shlib-compat.h>

# if LIB_COMPAT(libc, GLIBC_2_4, GLIBC_2_33)

/* Get information about the file FD in BUF.  */
int
__fxstatat (int vers, int fd, const char *file, struct stat *st, int flag)
{
#if STAT_IS_KERNEL_STAT
  /* New kABIs which uses generic pre 64-bit time Linux ABI, e.g.
     csky, nios2  */
  if (vers == _STAT_VER_KERNEL)
    {
      int r = INLINE_SYSCALL_CALL (fstatat64, fd, file, st, flag);
      return r ?: stat_overflow (st);
    }
  return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
#else
  /* Old kABIs with old non-LFS support, e.g. arm, i386, hppa, m68k, mips32,
     microblaze, s390, sh, powerpc32, and sparc32.  */
  struct stat64 st64;
  int r = INLINE_SYSCALL_CALL (fstatat64, fd, file, &st64, flag);
  return r ?: __xstat32_conv (vers, &st64, st);
#endif
}

# endif /* LIB_COMPAT  */

#endif /* XSTAT_IS_XSTAT64  */
