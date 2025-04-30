/* xstat using old-style Unix stat system call.
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

#include <sys/stat.h>
#include <fcntl.h>
#include <kernel_stat.h>
#include <sysdep.h>

#if !XSTAT_IS_XSTAT64
# include <xstatconv.h>
# include <xstatover.h>
# include <shlib-compat.h>

# if LIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_33)

/* Get information about the file NAME in BUF.  */
int
__xstat (int vers, const char *name, struct stat *buf)
{
  switch (vers)
    {
    case _STAT_VER_KERNEL:
      {
# if STAT_IS_KERNEL_STAT
	/* New kABIs which uses generic pre 64-bit time Linux ABI,
	   e.g. csky, nios2  */
	int r = INLINE_SYSCALL_CALL (fstatat64, AT_FDCWD, name, buf, 0);
	return r ?: stat_overflow (buf);
# else
	/* Old kABIs with old non-LFS support, e.g. arm, i386, hppa, m68k,
	   microblaze, s390, sh, powerpc, and sparc32.  */
	return INLINE_SYSCALL_CALL (stat, name, buf);
# endif
      }

    default:
      {
# if STAT_IS_KERNEL_STAT
	return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
# else
	struct stat64 buf64;
	int r = INLINE_SYSCALL_CALL (stat64, name, &buf64);
	return r ?: __xstat32_conv (vers, &buf64, buf);
#endif
      }
    }
}

# endif /* LIB_COMPAT  */

#endif /* XSTAT_IS_XSTAT64  */
