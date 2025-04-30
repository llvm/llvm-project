/* Machine-dependent ELF indirect relocation inline functions.
   HP-PARISC version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */

#ifndef _DL_IREL_H
#define _DL_IREL_H

#include <stdio.h>
#include <unistd.h>
#include <dl-fptr.h>

#define ELF_MACHINE_IREL	1

/* There is no IFUNC implementation yet. Add stubs to allow builds.  */

static inline struct fdesc
__attribute ((always_inline))
elf_ifunc_invoke (uintptr_t addr)
{
  return ((struct fdesc) {0, 0});
}

static inline void
__attribute ((always_inline))
elf_irel (const Elf32_Rel *reloc)
{
  return;
}

#endif /* dl-irel.h */
