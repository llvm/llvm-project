/* Machine-dependent ELF indirect relocation inline functions.
   SPARC 64-bit version.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#ifndef _DL_IREL_H
#define _DL_IREL_H

#include <stdio.h>
#include <unistd.h>
#include <dl-plt.h>
#include <ldsodefs.h>

#define ELF_MACHINE_IRELA	1

static inline Elf64_Addr
__attribute ((always_inline))
elf_ifunc_invoke (Elf64_Addr addr)
{
  return ((Elf64_Addr (*) (int)) (addr)) (GLRO(dl_hwcap));
}

static inline void
__attribute ((always_inline))
elf_irela (const Elf64_Rela *reloc)
{
  unsigned int r_type = (reloc->r_info & 0xff);

  if (__glibc_likely (r_type == R_SPARC_IRELATIVE))
    {
      Elf64_Addr *const reloc_addr = (void *) reloc->r_offset;
      Elf64_Addr value = elf_ifunc_invoke(reloc->r_addend);
      *reloc_addr = value;
    }
  else if (__glibc_likely (r_type == R_SPARC_JMP_IREL))
    {
      Elf64_Addr *const reloc_addr = (void *) reloc->r_offset;
      Elf64_Addr value = elf_ifunc_invoke(reloc->r_addend);
      struct link_map map = { .l_addr = 0 };

      /* 'high' is always zero, for large PLT entries the linker
	 emits an R_SPARC_IRELATIVE.  */
      sparc64_fixup_plt (&map, reloc, reloc_addr, value, 0, 0);
    }
  else if (r_type == R_SPARC_NONE)
    ;
  else
    __libc_fatal ("Unexpected reloc type in static binary.\n");
}

#endif /* dl-irel.h */
