/* Definitions for BSD-style memory management.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

/* These are the bits used by 4.4 BSD and its derivatives.  On systems
   (such as GNU) where these facilities are not system services but can be
   emulated in the C library, these are the definitions we emulate.  */

#ifndef _SYS_MMAN_H
# error "Never use <bits/mman.h> directly; include <sys/mman.h> instead."
#endif

/* Protections are chosen from these bits, OR'd together.  The
   implementation does not necessarily support PROT_EXEC or PROT_WRITE
   without PROT_READ.  The only guarantees are that no writing will be
   allowed without PROT_WRITE and no access will be allowed for PROT_NONE. */

#define	PROT_NONE	 0x00	/* No access.  */
#define	PROT_READ	 0x04	/* Pages can be read.  */
#define	PROT_WRITE	 0x02	/* Pages can be written.  */
#define	PROT_EXEC	 0x01	/* Pages can be executed.  */

/* Flags contain mapping type, sharing type and options.  */

/* Mapping type (must choose one and only one of these).  */
#ifdef __USE_MISC
# define MAP_FILE	 0x0001	/* Mapped from a file or device.  */
# define MAP_ANON	 0x0002	/* Allocated from anonymous virtual memory.  */
# define MAP_TYPE	 0x000f	/* Mask for type field.  */
# define MAP_ANONYMOUS	 MAP_ANON /* Linux name. */
#endif

/* Sharing types (must choose one and only one of these).  */
#ifdef __USE_MISC
# define MAP_COPY	 0x0020	/* Virtual copy of region at mapping time.  */
#endif
#define	MAP_SHARED	 0x0010	/* Share changes.  */
#define	MAP_PRIVATE	 0x0000	/* Changes private; copy pages on write.  */

/* Other flags.  */
#define	MAP_FIXED	 0x0100	/* Map address must be exactly as requested. */
#ifdef __USE_MISC
# define MAP_NOEXTEND	 0x0200	/* For MAP_FILE, don't change file size.  */
# define MAP_HASSEMPHORE 0x0400	/* Region may contain semaphores.  */
# define MAP_INHERIT	 0x0800	/* Region is retained after exec.  */
#endif

/* Advice to `madvise'.  */
#ifdef __USE_MISC
# define MADV_NORMAL	 0	/* No further special treatment.  */
# define MADV_RANDOM	 1	/* Expect random page references.  */
# define MADV_SEQUENTIAL 2	/* Expect sequential page references.  */
# define MADV_WILLNEED	 3	/* Will need these pages.  */
# define MADV_DONTNEED	 4	/* Don't need these pages.  */
#endif

/* The POSIX people had to invent similar names for the same things.  */
#ifdef __USE_XOPEN2K
# define POSIX_MADV_NORMAL	0 /* No further special treatment.  */
# define POSIX_MADV_RANDOM	1 /* Expect random page references.  */
# define POSIX_MADV_SEQUENTIAL	2 /* Expect sequential page references.  */
# define POSIX_MADV_WILLNEED	3 /* Will need these pages.  */
# define POSIX_MADV_DONTNEED	4 /* Don't need these pages.  */
#endif

/* Flags to `msync'.  */
#define MS_ASYNC	1		/* Sync memory asynchronously.  */
#define MS_SYNC		0		/* Synchronous memory sync.  */
#define MS_INVALIDATE	2		/* Invalidate the caches.  */

/* Flags for `mremap'.  */
#ifdef __USE_GNU
# define MREMAP_MAYMOVE	1		/* Mapping address may change.  */
# define MREMAP_FIXED	2		/* Fifth argument sets new address.  */
#endif

/* Flags for `mlockall' (can be OR'd together).  */
#define MCL_CURRENT	1		/* Lock all currently mapped pages.  */
#define MCL_FUTURE	2		/* Lock all additions to address
					   space.  */
