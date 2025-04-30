/* Definitions for POSIX memory map interface.  Linux/Alpha version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _SYS_MMAN_H
# error "Never use <bits/mman.h> directly; include <sys/mman.h> instead."
#endif

/* The following definitions basically come from the kernel headers.
   But the kernel header is not namespace clean.  */

#define __MAP_ANONYMOUS	  0x10		/* Don't use a file.  */

/* These are Linux-specific.  */
#ifdef __USE_MISC
# define MAP_GROWSDOWN	  0x01000	/* Stack-like segment.  */
# define MAP_DENYWRITE	  0x02000	/* ETXTBSY */
# define MAP_EXECUTABLE	  0x04000	/* Mark it as an executable.  */
# define MAP_LOCKED	  0x08000	/* Lock the mapping.  */
# define MAP_NORESERVE	  0x10000	/* Don't check for reservations.  */
# define MAP_POPULATE	  0x20000	/* Populate (prefault) pagetables.  */
# define MAP_NONBLOCK	  0x40000	/* Do not block on IO.  */
# define MAP_STACK	  0x80000	/* Allocation is for a stack.  */
# define MAP_HUGETLB	  0x100000	/* Create huge page mapping.  */
# define MAP_FIXED_NOREPLACE 0x200000	/* MAP_FIXED but do not unmap
					   underlying mapping.  */
#endif

/* Flags for `mlockall'.  */
#define MCL_CURRENT	  8192
#define MCL_FUTURE	  16384
#define MCL_ONFAULT	  32768

#include <bits/mman-linux.h>

/* Values that differ from standard <mman-linux.h>.  For the most part newer
   values are shared, but older values are skewed.  */

#undef  MAP_FIXED
#define MAP_FIXED	  0x100

#undef  MS_SYNC
#define MS_SYNC		  2
#undef  MS_INVALIDATE
#define MS_INVALIDATE	  4

#ifdef __USE_MISC
# undef  MADV_DONTNEED
# define MADV_DONTNEED    6
#endif
#ifdef __USE_XOPEN2K
# undef  POSIX_MADV_DONTNEED
# define POSIX_MADV_DONTNEED	6
#endif
