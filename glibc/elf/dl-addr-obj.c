/* Determine if address is inside object load segments.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <link.h>
#include <elf.h>

/* Return non-zero if ADDR lies within one of L's loadable segments.
   We have three cases we care about.

   Case 1: addr is above a segment.
   +==================+<- l_map_end
   |                  |<- addr
   |------------------|<- l_addr + p_vaddr + p_memsz
   |                  |
   |                  |
   |------------------|<- l_addr + p_vaddr
   |------------------|<- l_addr
   |                  |
   +==================+<- l_map_start

   Case 2: addr is within a segments.
   +==================+<- l_map_end
   |                  |
   |------------------|<- l_addr + p_vaddr + p_memsz
   |                  |<- addr
   |                  |
   |------------------|<- l_addr + p_vaddr
   |------------------|<- l_addr
   |                  |
   +==================+<- l_map_start

   Case 3: addr is below a segments.
   +==================+<- l_map_end
   |                  |
   |------------------|<- l_addr + p_vaddr + p_memsz
   |                  |
   |                  |
   |------------------|<- l_addr + p_vaddr
   |------------------|<- l_addr
   |                  |<- addr
   +==================+<- l_map_start

   All the arithmetic is unsigned and we shift all the values down by
   l_addr + p_vaddr and then compare the normalized addr to the range
   of interest i.e. 0 <= addr < p_memsz.

*/
int
_dl_addr_inside_object (struct link_map *l, const ElfW(Addr) addr)
{
  int n = l->l_phnum;
  const ElfW(Addr) reladdr = addr - l->l_addr;

  while (--n >= 0)
    if (l->l_phdr[n].p_type == PT_LOAD
	&& reladdr - l->l_phdr[n].p_vaddr < l->l_phdr[n].p_memsz)
      return 1;
  return 0;
}
