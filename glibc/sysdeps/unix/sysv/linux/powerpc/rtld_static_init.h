/* Partial initialization of ld.so loaded via static dlopen.  powerpc helper.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

static inline void
__rtld_static_init_arch (struct link_map *map, struct rtld_global_ro *dl)
{
  /* This field does not exist in the generic _rtld_global_ro version.  */

  extern __typeof (dl->_dl_cache_line_size) _dl_cache_line_size
    attribute_hidden;
  dl->_dl_cache_line_size = _dl_cache_line_size;
}
