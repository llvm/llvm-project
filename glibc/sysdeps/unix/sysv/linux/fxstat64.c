/* fxstat64 using Linux fstat64/statx system call.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#define __fxstat __redirect___fxstat
#include <sys/stat.h>
#undef __fxstat
#include <fcntl.h>
#include <kernel_stat.h>
#include <sysdep.h>
#include <xstatconv.h>
#include <statx_cp.h>
#include <shlib-compat.h>

#if LIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_33)

/* Get information about the file FD in BUF.  */

int
___fxstat64 (int vers, int fd, struct stat64 *buf)
{
#if XSTAT_IS_XSTAT64
# ifdef __NR_fstat64
  /* 64-bit kABI outlier, e.g. sparc64.  */
  if (vers == _STAT_VER_KERNEL)
    return INLINE_SYSCALL_CALL (fstat, fd, buf);
  else
    {
      struct stat64 st64;
      int r = INLINE_SYSCALL_CALL (fstat64, fd, &st64);
      return r ?: __xstat32_conv (vers, &st64, (struct stat *) buf);
    }
# elif defined __NR_fstat
  /* 64-bit kABI, e.g. aarch64, ia64, powerpc64*, s390x, riscv64,
     and x86_64.  */
  if (vers == _STAT_VER_KERNEL || vers == _STAT_VER_LINUX)
    return INLINE_SYSCALL_CALL (fstat, fd, buf);
  return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
# else
  /* New 32-bit kABIs with only 64-bit time_t support, e.g. arc, riscv32.  */
  struct statx tmp;
  int r = INLINE_SYSCALL_CALL (statx, fd, "", AT_EMPTY_PATH,
			       STATX_BASIC_STATS, &tmp);
  if (r == 0)
    __cp_stat64_statx (buf, &tmp);
  return r;
# endif
#else
  /* All kABIs with non-LFS support, e.g. arm, csky, i386, hppa, m68k,
     microblaze, mips32, nios2, sh, powerpc32, and sparc32.  */
  return INLINE_SYSCALL_CALL (fstat64, fd, buf);
#endif /* XSTAT_IS_XSTAT64  */
}

#if SHLIB_COMPAT(libc, GLIBC_2_1, GLIBC_2_2)
versioned_symbol (libc, ___fxstat64, __fxstat64, GLIBC_2_2);
strong_alias (___fxstat64, __old__fxstat64)
compat_symbol (libc, __old__fxstat64, __fxstat64, GLIBC_2_1);
#else
strong_alias (___fxstat64, __fxstat64)
#endif

#if XSTAT_IS_XSTAT64
strong_alias (___fxstat64, __fxstat)
#endif

#endif /* LIB_COMPAT  */
