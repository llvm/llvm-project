/* MIPS-specific handling of undefined symbols.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _ELF_MACHINE_SYM_NO_MATCH_H
#define _ELF_MACHINE_SYM_NO_MATCH_H

#include <link.h>
#include <stdbool.h>

/* The semantics of zero/non-zero values of undefined symbols differs
   depending on whether the non-PIC ABI is in use.  Under the non-PIC
   ABI, a non-zero value indicates that there is an address reference
   to the symbol and thus it must always be resolved (except when
   resolving a jump slot relocation) to the PLT entry whose address is
   provided as the symbol's value; a zero value indicates that this
   canonical-address behaviour is not required.  Yet under the classic
   MIPS psABI, a zero value indicates that there is an address
   reference to the function and the dynamic linker must resolve the
   symbol immediately upon loading.  To avoid conflict, symbols for
   which the dynamic linker must assume the non-PIC ABI semantics are
   marked with the STO_MIPS_PLT flag.  */
static inline bool
elf_machine_sym_no_match (const ElfW(Sym) *sym)
{
  return sym->st_shndx == SHN_UNDEF && !(sym->st_other & STO_MIPS_PLT);
}

#endif /* _ELF_MACHINE_SYM_NO_MATCH_H */
