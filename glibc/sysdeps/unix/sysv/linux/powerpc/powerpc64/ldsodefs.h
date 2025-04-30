/* Run-time dynamic linker data structures for loaded ELF shared objects.
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

#ifndef	_LDSODEFS_H

/* Get the real definitions.  */
#include_next <ldsodefs.h>

/* Now define our stuff.  */

#if _CALL_ELF != 2

static __always_inline bool
_dl_ppc64_is_opd_sym (const struct link_map *l, const ElfW(Sym) *sym)
{
  return (ELFW(ST_TYPE) (sym->st_info) == STT_FUNC
	  && l->l_addr + sym->st_value >= (ElfW(Addr)) l->l_ld
	  && l->l_addr + sym->st_value < l->l_map_end
	  && sym->st_size != 0);
}

static __always_inline bool
_dl_ppc64_addr_sym_match (const struct link_map *l, const ElfW(Sym) *sym,
			  const ElfW(Sym) *matchsym, ElfW(Addr) addr)
{
  ElfW(Addr) value = l->l_addr + sym->st_value;
  if (_dl_ppc64_is_opd_sym (l, sym))
    {
      if (addr < value || addr >= value + 24)
	{
	  value = *(ElfW(Addr) *) value;
	  if (addr < value || addr >= value + sym->st_size)
	    return false;
	}
    }
  else if (sym->st_shndx == SHN_UNDEF || sym->st_size == 0)
    {
      if (addr != value)
	return false;
    }
  else if (addr < value || addr >= value + sym->st_size)
    return false;

  if (matchsym == NULL)
    return true;

  ElfW(Addr) matchvalue = l->l_addr + matchsym->st_value;
  if (_dl_ppc64_is_opd_sym (l, matchsym)
      && (addr < matchvalue || addr > matchvalue + 24))
    matchvalue = *(ElfW(Addr) *) matchvalue;

  return matchvalue < value;
}

/* If this is a function symbol defined past the end of our dynamic
   section, then it must be a function descriptor.  Allow these symbols
   to match their associated function code range as well as the
   descriptor addresses.  */
#undef DL_ADDR_SYM_MATCH
#define DL_ADDR_SYM_MATCH(L, SYM, MATCHSYM, ADDR) \
  _dl_ppc64_addr_sym_match (L, SYM, MATCHSYM, ADDR)

#endif

#endif /* ldsodefs.h */
