/* mmap - map files or devices into memory.  Linux version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 1999.

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
#include <mmap_internal.h>

#ifdef __NR_mmap2
/* To avoid silent truncation of offset when using mmap2, do not accept
   offset larger than 1 << (page_shift + off_t bits).  For archictures with
   32 bits off_t and page size of 4096 it would be 1^44.  */
# define MMAP_OFF_HIGH_MASK \
  ((-(MMAP2_PAGE_UNIT << 1) << (8 * sizeof (off_t) - 1)))
#else
/* Some ABIs might use __NR_mmap while having sizeof (off_t) smaller than
   sizeof (off64_t) (currently only MIPS64n32).  For this case just set
   zero the higher bits so mmap with large offset does not fail.  */
# define MMAP_OFF_HIGH_MASK  0x0
#endif

#define MMAP_OFF_MASK (MMAP_OFF_HIGH_MASK | MMAP_OFF_LOW_MASK)

/* An architecture may override this.  */
#ifndef MMAP_PREPARE
# define MMAP_PREPARE(addr, len, prot, flags, fd, offset)
#endif

__attribute__((noinline))
void *
__mmap_internal (void *addr, size_t len, int prot, int flags, int fd, off64_t offset)
{
  MMAP_CHECK_PAGE_UNIT ();

  if (offset & MMAP_OFF_MASK)
    return (void *) INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);

  MMAP_PREPARE (addr, len, prot, flags, fd, offset);
#ifdef __NR_mmap2
  return (void *) MMAP_CALL (mmap2, addr, len, prot, flags, fd,
			     (off_t) (offset / MMAP2_PAGE_UNIT));
#else
  return (void *) MMAP_CALL (mmap, addr, len, prot, flags, fd, offset);
#endif
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
void *__mmap_nextsilicon(void *start, size_t len, int prot, int flags, int fd, off_t off)
{
	return __mmap_internal(start, len, prot, flags, fd, off);
}

void *__mmap64(void *start, size_t len, int prot, int flags, int fd, off_t off)
{
	return __mmap_nextsilicon(start, len, prot, flags, fd, off);
}

libc_hidden_def (__mmap_internal)

weak_alias (__mmap64, mmap64)
libc_hidden_def (__mmap64)

#ifdef __OFF_T_MATCHES_OFF64_T
weak_alias (__mmap64, mmap)
weak_alias (__mmap64, __mmap)
libc_hidden_def (__mmap)
#endif
