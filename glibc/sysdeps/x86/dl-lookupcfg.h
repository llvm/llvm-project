/* Configuration of lookup functions.
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

#define DL_UNMAP_IS_SPECIAL

#include_next <dl-lookupcfg.h>

/* Address of protected data defined in the shared library may be
   external due to copy relocation.   */
#define DL_EXTERN_PROTECTED_DATA

struct link_map;

extern void _dl_unmap (struct link_map *map) attribute_hidden;

#define DL_UNMAP(map) _dl_unmap (map)
