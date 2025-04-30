/* Definitions for POSIX memory map interface.  Linux/HPPA version.
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

/* These are taken from the kernel definitions.  */

/* Other flags.  */
#define __MAP_ANONYMOUS	0x10		/* Don't use a file */
#ifdef __USE_MISC
# define MAP_VARIABLE	0
#endif

/* These are Linux-specific.  */
#ifdef __USE_MISC
# define MAP_DENYWRITE	0x0800		/* ETXTBSY */
# define MAP_EXECUTABLE	0x1000		/* Mark it as an executable */
# define MAP_LOCKED	0x2000		/* Pages are locked */
# define MAP_NORESERVE	0x4000		/* Don't check for reservations */
# define MAP_GROWSDOWN	0x8000		/* Stack-like segment */
# define MAP_POPULATE	0x10000		/* Populate (prefault) pagetables */
# define MAP_NONBLOCK	0x20000		/* Do not block on IO */
# define MAP_STACK	0x40000		/* Create for process/thread stacks */
# define MAP_HUGETLB	0x80000		/* Create a huge page mapping */
# define MAP_FIXED_NOREPLACE 0x100000	/* MAP_FIXED but do not unmap
					   underlying mapping.  */
#endif

/* Advice to "madvise"  */
#ifdef __USE_MISC
# define MADV_SOFT_OFFLINE 101	/* Soft offline page for testing.  */
#endif

#include <bits/mman-linux.h>

#ifdef __USE_MISC
# undef MAP_TYPE
# define MAP_TYPE	0x2b		/* Mask for type of mapping */
#endif

#undef MAP_FIXED
#define MAP_FIXED	0x04		/* Interpret addr exactly */

/* Flags to "msync"  */
#undef MS_SYNC
#define MS_SYNC		1		/* Synchronous memory sync */
#undef MS_ASYNC
#define MS_ASYNC	2		/* Sync memory asynchronously */
#undef MS_INVALIDATE
#define MS_INVALIDATE	4		/* Invalidate the caches */

/* Advice to "madvise"  */
#ifdef __USE_MISC
# undef MADV_MERGEABLE
# define MADV_MERGEABLE   65	/* KSM may merge identical pages */
# undef MADV_UNMERGEABLE
# define MADV_UNMERGEABLE 66	/* KSM may not merge identical pages */
# undef MADV_HUGEPAGE
# define MADV_HUGEPAGE	 67	/* Worth backing with hugepages */
# undef MADV_NOHUGEPAGE
# define MADV_NOHUGEPAGE 68	/* Not worth backing with hugepages */
# undef MADV_DONTDUMP
# define MADV_DONTDUMP	 69	/* Explicity exclude from the core dump,
				   overrides the coredump filter bits */
# undef MADV_DODUMP
# define MADV_DODUMP	 70	/* Clear the MADV_NODUMP flag */
# undef MADV_WIPEONFORK
# define MADV_WIPEONFORK 71	/* Zero memory on fork, child only.  */
# undef MADV_KEEPONFORK
# define MADV_KEEPONFORK 72	/* Undo MADV_WIPEONFORK.  */
#endif
