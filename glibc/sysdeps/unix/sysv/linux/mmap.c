/* mmap - map files or devices into memory.  Linux version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <unistd.h>
#include <sys/mman.h>
#include <sysdep.h>
#include <stdint.h>

#ifndef __OFF_T_MATCHES_OFF64_T
# include <mmap_internal.h>

/* An architecture may override this.  */
# ifndef MMAP_ADJUST_OFFSET
#  define MMAP_ADJUST_OFFSET(offset) offset
# endif

void *
__mmap (void *addr, size_t len, int prot, int flags, int fd, off_t offset)
{
  MMAP_CHECK_PAGE_UNIT ();

  if (offset & MMAP_OFF_LOW_MASK)
    return (void *) INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);

#ifdef __NR_mmap2
  return (void *) MMAP_CALL (mmap2, addr, len, prot, flags, fd,
			     offset / (uint32_t) MMAP2_PAGE_UNIT);
#else
  return (void *) MMAP_CALL (mmap, addr, len, prot, flags, fd,
			     MMAP_ADJUST_OFFSET (offset));
#endif
}
weak_alias (__mmap, mmap)
libc_hidden_def (__mmap)

#endif /* __OFF_T_MATCHES_OFF64_T  */
