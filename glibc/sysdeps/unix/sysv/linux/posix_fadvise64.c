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
#include <shlib-compat.h>

int __posix_fadvise64_l64 (int fd, off64_t offset, off64_t len, int advise);
libc_hidden_proto (__posix_fadvise64_l64)

/* Both arm and powerpc implements fadvise64_64 with last 'advise' argument
   just after 'fd' to avoid the requirement of implementing 7-arg syscalls.
   ARM also defines __NR_fadvise64_64 as __NR_arm_fadvise64_64.

   s390 implements fadvice64_64 using a specific struct with arguments
   packed inside.  This is the only implementation handled in arch-specific
   code.  */

#ifndef __NR_fadvise64_64
# define __NR_fadvise64_64 __NR_fadvise64
#endif

/* Advice the system about the expected behaviour of the application with
   respect to the file associated with FD.  */

int
__posix_fadvise64_l64 (int fd, off64_t offset, off64_t len, int advise)
{
#ifdef __ASSUME_FADVISE64_64_6ARG
  int ret = INTERNAL_SYSCALL_CALL (fadvise64_64, fd, advise,
				   SYSCALL_LL64 (offset), SYSCALL_LL64 (len));
#else
  int ret = INTERNAL_SYSCALL_CALL (fadvise64_64, fd,
				   __ALIGNMENT_ARG SYSCALL_LL64 (offset),
				   SYSCALL_LL64 (len), advise);
#endif
  if (!INTERNAL_SYSCALL_ERROR_P (ret))
    return 0;
  return INTERNAL_SYSCALL_ERRNO (ret);
}

/* The type of the len argument was changed from size_t to off_t in
   POSIX.1-2003 TC1.  */
#ifndef __OFF_T_MATCHES_OFF64_T
# if SHLIB_COMPAT(libc, GLIBC_2_2, GLIBC_2_3_3)
int __posix_fadvise64_l32 (int fd, off64_t offset, size_t len, int advise);

int
attribute_compat_text_section
__posix_fadvise64_l32 (int fd, off64_t offset, size_t len, int advise)
{
  return __posix_fadvise64_l64 (fd, offset, len, advise);
}

versioned_symbol (libc, __posix_fadvise64_l64, posix_fadvise64, GLIBC_2_3_3);
compat_symbol (libc, __posix_fadvise64_l32, posix_fadvise64, GLIBC_2_2);
# else
weak_alias (__posix_fadvise64_l64, posix_fadvise64);
# endif
#else
weak_alias (__posix_fadvise64_l64, posix_fadvise64);
strong_alias (__posix_fadvise64_l64, posix_fadvise);
#endif
libc_hidden_def (__posix_fadvise64_l64)
