/* Locate the shared object symbol nearest a given address.
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

#include <dlfcn.h>
#include <stddef.h>
#include <ldsodefs.h>


static inline void
__attribute ((always_inline))
determine_info (const ElfW(Addr) addr, struct link_map *match, Dl_info *info,
		struct link_map **mapp, const ElfW(Sym) **symbolp)
{
  /* Now we know what object the address lies in.  */
  info->dli_fname = match->l_name;
  info->dli_fbase = (void *) match->l_map_start;

  /* If this is the main program the information is incomplete.  */
  if (__builtin_expect (match->l_name[0], 'a') == '\0'
      && match->l_type == lt_executable)
    info->dli_fname = _dl_argv[0];

  const ElfW(Sym) *symtab
    = (const ElfW(Sym) *) D_PTR (match, l_info[DT_SYMTAB]);
  const char *strtab = (const char *) D_PTR (match, l_info[DT_STRTAB]);

  ElfW(Word) strtabsize = match->l_info[DT_STRSZ]->d_un.d_val;

  const ElfW(Sym) *matchsym = NULL;
  if (match->l_info[ELF_MACHINE_GNU_HASH_ADDRIDX] != NULL)
    {
      /* We look at all symbol table entries referenced by the hash
	 table.  */
      for (Elf_Symndx bucket = 0; bucket < match->l_nbuckets; ++bucket)
	{
	  Elf32_Word symndx = match->l_gnu_buckets[bucket];
	  if (symndx != 0)
	    {
	      const Elf32_Word *hasharr = &match->l_gnu_chain_zero[symndx];

	      do
		{
		  /* The hash table never references local symbols so
		     we can omit that test here.  */
		  symndx = ELF_MACHINE_HASH_SYMIDX (match, hasharr);
		  if ((symtab[symndx].st_shndx != SHN_UNDEF
		       || symtab[symndx].st_value != 0)
		      && symtab[symndx].st_shndx != SHN_ABS
		      && ELFW(ST_TYPE) (symtab[symndx].st_info) != STT_TLS
		      && DL_ADDR_SYM_MATCH (match, &symtab[symndx],
					    matchsym, addr)
		      && symtab[symndx].st_name < strtabsize)
		    matchsym = (ElfW(Sym) *) &symtab[symndx];
		}
	      while ((*hasharr++ & 1u) == 0);
	    }
	}
    }
  else
    {
      const ElfW(Sym) *symtabend;
      if (match->l_info[DT_HASH] != NULL)
	symtabend = (symtab
		     + ((Elf_Symndx *) D_PTR (match, l_info[DT_HASH]))[1]);
      else
	/* There is no direct way to determine the number of symbols in the
	   dynamic symbol table and no hash table is present.  The ELF
	   binary is ill-formed but what shall we do?  Use the beginning of
	   the string table which generally follows the symbol table.  */
	symtabend = (const ElfW(Sym) *) strtab;

      for (; (void *) symtab < (void *) symtabend; ++symtab)
	if ((ELFW(ST_BIND) (symtab->st_info) == STB_GLOBAL
	     || ELFW(ST_BIND) (symtab->st_info) == STB_WEAK)
	    && __glibc_likely (!dl_symbol_visibility_binds_local_p (symtab))
	    && ELFW(ST_TYPE) (symtab->st_info) != STT_TLS
	    && (symtab->st_shndx != SHN_UNDEF
		|| symtab->st_value != 0)
	    && symtab->st_shndx != SHN_ABS
	    && DL_ADDR_SYM_MATCH (match, symtab, matchsym, addr)
	    && symtab->st_name < strtabsize)
	  matchsym = (ElfW(Sym) *) symtab;
    }

  if (mapp)
    *mapp = match;
  if (symbolp)
    *symbolp = matchsym;

  if (matchsym)
    {
      /* We found a symbol close by.  Fill in its name and exact
	 address.  */
      lookup_t matchl = LOOKUP_VALUE (match);

      info->dli_sname = strtab + matchsym->st_name;
      info->dli_saddr = DL_SYMBOL_ADDRESS (matchl, matchsym);
    }
  else
    {
      /* No symbol matches.  We return only the containing object.  */
      info->dli_sname = NULL;
      info->dli_saddr = NULL;
    }
}


int
_dl_addr (const void *address, Dl_info *info,
	  struct link_map **mapp, const ElfW(Sym) **symbolp)
{
  const ElfW(Addr) addr = DL_LOOKUP_ADDRESS (address);
  int result = 0;

  /* Protect against concurrent loads and unloads.  */
  __rtld_lock_lock_recursive (GL(dl_load_lock));

  struct link_map *l = _dl_find_dso_for_object (addr);

  if (l)
    {
      determine_info (addr, l, info, mapp, symbolp);
      result = 1;
    }

  __rtld_lock_unlock_recursive (GL(dl_load_lock));

  return result;
}
