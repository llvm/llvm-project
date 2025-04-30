/* On-demand PLT fixup for shared objects.  HPPA version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <elf/dl-runtime.c>

/* The caller has encountered a partially relocated function descriptor.
   The gp of the descriptor has been updated, but not the ip.  We find
   the function descriptor again and compute the relocation offset and
   return that to the caller.  The caller will continue on to call
   _dl_fixup with the relocation offset.  */

ElfW(Word)
attribute_hidden __attribute ((noinline)) ARCH_FIXUP_ATTRIBUTE
_dl_fix_reloc_arg (struct fdesc *fptr, struct link_map *l)
{
  Elf32_Addr l_addr, iplt, jmprel, end_jmprel, r_type;
  const Elf32_Rela *reloc;

  l_addr = l->l_addr;
  jmprel = D_PTR(l, l_info[DT_JMPREL]);
  end_jmprel = jmprel + l->l_info[DT_PLTRELSZ]->d_un.d_val;

  /* Look for the entry...  */
  for (iplt = jmprel; iplt < end_jmprel; iplt += sizeof (Elf32_Rela))
    {
      reloc = (const Elf32_Rela *) iplt;
      r_type = ELF32_R_TYPE (reloc->r_info);

      if (__builtin_expect (r_type == R_PARISC_IPLT, 1)
	  && fptr == (struct fdesc *) (reloc->r_offset + l_addr))
	/* Found entry. Return the reloc offset.  */
	return iplt - jmprel;
    }

  /* Crash if we weren't passed a valid function pointer.  */
  ABORT_INSTRUCTION;
  return 0;
}
