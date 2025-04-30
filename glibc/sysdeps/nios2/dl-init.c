/* Nios II specific procedures for initializing code.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
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

#include <elf/dl-init.c>

unsigned int
_dl_nios2_get_gp_value (struct link_map *main_map)
{
  ElfW(Dyn) *dyn = main_map->l_ld;
  for (dyn = main_map->l_ld; dyn->d_tag != DT_NULL; ++dyn)
    if (dyn->d_tag == DT_NIOS2_GP)
      return (unsigned int)(dyn->d_un.d_ptr);
  return 0;
}
