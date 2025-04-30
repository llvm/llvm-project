/* Machine-dependent ELF indirect relocation inline functions.
   ARM version.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#ifndef _DL_IREL_H
#define _DL_IREL_H

#include <stdio.h>
#include <unistd.h>
#include <ldsodefs.h>

#define ELF_MACHINE_IREL	1

static inline Elf32_Addr
__attribute ((always_inline))
elf_ifunc_invoke (Elf32_Addr addr)
{
  return ((Elf32_Addr (*) (unsigned long int)) (addr)) (GLRO(dl_hwcap));
}

static inline void
__attribute ((always_inline))
elf_irel (const Elf32_Rel *reloc)
{
  Elf32_Addr *const reloc_addr = (void *) reloc->r_offset;
  const unsigned long int r_type = ELF32_R_TYPE (reloc->r_info);

  if (__builtin_expect (r_type == R_ARM_IRELATIVE, 1))
    {
      Elf32_Addr value = elf_ifunc_invoke (*reloc_addr);
      *reloc_addr = value;
    }
  else
    __libc_fatal ("Unexpected reloc type in static binary.\n");
}

#endif /* dl-irel.h */
