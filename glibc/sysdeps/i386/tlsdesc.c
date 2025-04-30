/* Manage TLS descriptors.  i386 version.
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

#include <ldsodefs.h>
#include <tls.h>
#include <dl-tlsdesc.h>
#include <dl-unmap-segments.h>
#include <tlsdeschtab.h>

/* Unmap the dynamic object, but also release its TLS descriptor table
   if there is one.  */

void
_dl_unmap (struct link_map *map)
{
  _dl_unmap_segments (map);

#ifdef SHARED
  if (map->l_mach.tlsdesc_table)
    htab_delete (map->l_mach.tlsdesc_table);
#endif
}
