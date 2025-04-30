/* ELF symbol resolve functions for VDSO objects.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#ifndef _DL_VDSO_H
#define _DL_VDSO_H	1

#include <ldsodefs.h>
#include <dl-hash.h>

/* If the architecture support vDSO it should define which is the expected
   kernel version and hash value through both VDSO_NAME and VDSO_HASH
   (usually defined at architecture sysdep.h).  */

#ifndef VDSO_NAME
# define VDSO_NAME "LINUX_0.0"
#endif
#ifndef VDSO_HASH
# define VDSO_HASH 0
#endif

/* Functions for resolving symbols in the VDSO link map.  */
static inline void *
dl_vdso_vsym (const char *name)
{
  struct link_map *map = GLRO (dl_sysinfo_map);
  if (map == NULL)
    return NULL;

  /* Use a WEAK REF so we don't error out if the symbol is not found.  */
  ElfW (Sym) wsym = { 0 };
  wsym.st_info = (unsigned char) ELFW (ST_INFO (STB_WEAK, STT_NOTYPE));

  struct r_found_version rfv = { VDSO_NAME, VDSO_HASH, 1, NULL };

  /* Search the scope of the vdso map.  */
  const ElfW (Sym) *ref = &wsym;
  lookup_t result = GLRO (dl_lookup_symbol_x) (name, map, &ref,
					       map->l_local_scope,
					       &rfv, 0, 0, NULL);
  return ref != NULL ? DL_SYMBOL_ADDRESS (result, ref) : NULL;
}

#endif /* dl-vdso.h */
