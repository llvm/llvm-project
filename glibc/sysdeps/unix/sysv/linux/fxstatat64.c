/* fxstatat64 used on fstatat64, Linux implementation.
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

#define __fxstatat __redirect___fxstatat
#include <sys/stat.h>
#undef __fxstatat
#include <fcntl.h>
#include <kernel_stat.h>
#include <sysdep.h>
#include <xstatconv.h>
#include <statx_cp.h>
#include <shlib-compat.h>

#if LIB_COMPAT(libc, GLIBC_2_4, GLIBC_2_33)

/* Get information about the file FD in BUF.  */

int
__fxstatat64 (int vers, int fd, const char *file, struct stat64 *st, int flag)
{
#if XSTAT_IS_XSTAT64
# ifdef __NR_newfstatat
  /* 64-bit kABI, e.g. aarch64, ia64, powerpc64*, s390x, riscv64, and
     x86_64.  */
  if (vers == _STAT_VER_KERNEL || vers == _STAT_VER_LINUX)
    return INLINE_SYSCALL_CALL (newfstatat, fd, file, st, flag);
# elif defined __NR_fstatat64
  /* 64-bit kABI outlier, e.g. sparc64.  */
  struct stat64 st64;
  int r = INLINE_SYSCALL_CALL (fstatat64, fd, file, &st64, flag);
  return r ?: __xstat32_conv (vers, &st64, (struct stat *) st);
# else
  /* New 32-bit kABIs with only 64-bit time_t support, e.g. arc, riscv32.  */
  if (vers == _STAT_VER_KERNEL)
    {
      struct statx tmp;
      int r = INLINE_SYSCALL_CALL (statx, fd, file, AT_NO_AUTOMOUNT | flag,
				   STATX_BASIC_STATS, &tmp);
      if (r == 0)
	__cp_stat64_statx (st, &tmp);
      return r;
    }
# endif
#else
  /* All kABIs with non-LFS support, e.g. arm, csky, i386, hppa, m68k,
     microblaze, mips32, nios2, sh, powerpc32, and sparc32.  */
  if (vers == _STAT_VER_LINUX)
    return INLINE_SYSCALL_CALL (fstatat64, fd, file, st, flag);
#endif
  return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
}

#if XSTAT_IS_XSTAT64
strong_alias (__fxstatat64, __fxstatat)
#endif

#endif /* LIB_COMPAT(libc, GLIBC_2_4, GLIBC_2_33)  */
