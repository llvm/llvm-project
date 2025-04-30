/* Unmap a shared object's segments.  Generic version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _DL_UNMAP_SEGMENTS_H
#define _DL_UNMAP_SEGMENTS_H	1

#include <link.h>
#include <sys/mman.h>

#if IS_IN(rtld)
# define maybe_munmap_hook __munmap_hook
extern typeof (&__munmap) volatile __munmap_hook;
#else
# define maybe_munmap_hook __munmap
#endif

/* _dl_map_segments ensures that any whole pages in gaps between segments
   are filled in with PROT_NONE mappings.  So we can just unmap the whole
   range in one fell swoop.  */

static __always_inline void
_dl_unmap_segments (struct link_map *l)
{
  maybe_munmap_hook ((void *) l->l_map_start,
		     l->l_map_end - l->l_map_start);
}

#endif  /* dl-unmap-segments.h */
