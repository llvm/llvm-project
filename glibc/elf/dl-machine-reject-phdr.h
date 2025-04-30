/* Machine-dependent program header inspection for the ELF loader.
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

#ifndef _DL_MACHINE_REJECT_PHDR_H
#define _DL_MACHINE_REJECT_PHDR_H 1

#include <stdbool.h>

/* Return true iff ELF program headers are incompatible with the running
   host.  */
static inline bool
elf_machine_reject_phdr_p (const ElfW(Phdr) *phdr, uint_fast16_t phnum,
			   const char *buf, size_t len, struct link_map *map,
			   int fd)
{
  return false;
}

#endif /* dl-machine-reject-phdr.h */
