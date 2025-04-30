/* Support for GNU properties.  Generic version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#ifndef _DL_PROP_H
#define _DL_PROP_H

/* The following functions are used by the dynamic loader and the
   dlopen machinery to process PT_NOTE and PT_GNU_PROPERTY entries in
   the binary or shared object.  The notes can be used to change the
   behaviour of the loader, and as such offer a flexible mechanism
   for hooking in various checks related to ABI tags or implementing
   "flag day" ABI transitions.  */

static inline void __attribute__ ((always_inline))
_rtld_main_check (struct link_map *m, const char *program)
{
}

static inline void __attribute__ ((always_inline))
_dl_open_check (struct link_map *m)
{
}

static inline void __attribute__ ((always_inline))
_dl_process_pt_note (struct link_map *l, int fd, const ElfW(Phdr) *ph)
{
}

/* Called for each property in the NT_GNU_PROPERTY_TYPE_0 note of L,
   processing of the properties continues until this returns 0.  */
static inline int __attribute__ ((always_inline))
_dl_process_gnu_property (struct link_map *l, int fd, uint32_t type,
			  uint32_t datasz, void *data)
{
  return 0;
}

#endif /* _DL_PROP_H */
