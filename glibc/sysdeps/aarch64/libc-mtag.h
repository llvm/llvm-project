/* libc-internal interface for tagged (colored) memory support.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#ifndef _AARCH64_LIBC_MTAG_H
#define _AARCH64_LIBC_MTAG_H 1

#ifndef USE_MTAG
/* Generic bindings for systems that do not support memory tagging.  */
#include_next "libc-mtag.h"
#else

/* Used to ensure additional alignment when objects need to have distinct
   tags.  */
#define __MTAG_GRANULE_SIZE 16

/* Non-zero if memory obtained via morecore (sbrk) is not tagged.  */
#define __MTAG_SBRK_UNTAGGED 1

/* Extra flags to pass to mmap to get tagged pages.  */
#define __MTAG_MMAP_FLAGS PROT_MTE

/* Set the tags for a region of memory, which must have size and alignment
   that are multiples of __MTAG_GRANULE_SIZE.  Size cannot be zero.  */
void *__libc_mtag_tag_region (void *, size_t);

/* Optimized equivalent to __libc_mtag_tag_region followed by memset to 0.  */
void *__libc_mtag_tag_zero_region (void *, size_t);

/* Convert address P to a pointer that is tagged correctly for that
   location.  */
static __always_inline void *
__libc_mtag_address_get_tag (void *p)
{
  register void *x0 asm ("x0") = p;
  asm (".inst 0xd9600000 /* ldg x0, [x0] */" : "+r" (x0));
  return x0;
}

/* Assign a new (random) tag to a pointer P (does not adjust the tag on
   the memory addressed).  */
static __always_inline void *
__libc_mtag_new_tag (void *p)
{
  register void *x0 asm ("x0") = p;
  register unsigned long x1 asm ("x1");
  /* Guarantee that the new tag is not the same as now.  */
  asm (".inst 0x9adf1401 /* gmi x1, x0, xzr */\n"
       ".inst 0x9ac11000 /* irg x0, x0, x1 */" : "+r" (x0), "=r" (x1));
  return x0;
}

#endif /* USE_MTAG */

#endif /* _AARCH64_LIBC_MTAG_H */
