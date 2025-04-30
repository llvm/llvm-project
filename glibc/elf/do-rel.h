/* Do relocations for ELF dynamic linking.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

/* This file may be included twice, to define both
   `elf_dynamic_do_rel' and `elf_dynamic_do_rela'.  */

#ifdef DO_RELA
# define elf_dynamic_do_Rel		elf_dynamic_do_Rela
# define Rel				Rela
# define elf_machine_rel		elf_machine_rela
# define elf_machine_rel_relative	elf_machine_rela_relative
#endif

#ifndef DO_ELF_MACHINE_REL_RELATIVE
# define DO_ELF_MACHINE_REL_RELATIVE(map, l_addr, relative) \
  elf_machine_rel_relative (l_addr, relative,				      \
			    (void *) (l_addr + relative->r_offset))
#endif

/* Perform the relocations in MAP on the running program image as specified
   by RELTAG, SZTAG.  If LAZY is nonzero, this is the first pass on PLT
   relocations; they should be set up to call _dl_runtime_resolve, rather
   than fully resolved now.  */

auto inline void __attribute__ ((always_inline))
elf_dynamic_do_Rel (struct link_map *map,
		    ElfW(Addr) reladdr, ElfW(Addr) relsize,
		    __typeof (((ElfW(Dyn) *) 0)->d_un.d_val) nrelative,
		    int lazy, int skip_ifunc
#ifndef NESTING
		    , struct link_map *boot_map
#endif
		    )
{
  const ElfW(Rel) *r = (const void *) reladdr;
  const ElfW(Rel) *end = (const void *) (reladdr + relsize);
  ElfW(Addr) l_addr = map->l_addr;
# if defined ELF_MACHINE_IRELATIVE && !defined RTLD_BOOTSTRAP
  const ElfW(Rel) *r2 = NULL;
  const ElfW(Rel) *end2 = NULL;
# endif

#if (!defined DO_RELA || !defined ELF_MACHINE_PLT_REL) && !defined RTLD_BOOTSTRAP
  /* We never bind lazily during ld.so bootstrap.  Unfortunately gcc is
     not clever enough to see through all the function calls to realize
     that.  */
  if (lazy)
    {
      /* Doing lazy PLT relocations; they need very little info.  */
      for (; r < end; ++r)
# ifdef ELF_MACHINE_IRELATIVE
	if (ELFW(R_TYPE) (r->r_info) == ELF_MACHINE_IRELATIVE)
	  {
	    if (r2 == NULL)
	      r2 = r;
	    end2 = r;
	  }
	else
# endif
	  elf_machine_lazy_rel (map, l_addr, r, skip_ifunc);

# ifdef ELF_MACHINE_IRELATIVE
      if (r2 != NULL)
	for (; r2 <= end2; ++r2)
	  if (ELFW(R_TYPE) (r2->r_info) == ELF_MACHINE_IRELATIVE)
	    elf_machine_lazy_rel (map, l_addr, r2, skip_ifunc);
# endif
    }
  else
#endif
    {
      const ElfW(Sym) *const symtab =
	(const void *) D_PTR (map, l_info[DT_SYMTAB]);
      const ElfW(Rel) *relative = r;
      r += nrelative;

#ifndef RTLD_BOOTSTRAP
      /* This is defined in rtld.c, but nowhere in the static libc.a; make
	 the reference weak so static programs can still link.  This
	 declaration cannot be done when compiling rtld.c (i.e. #ifdef
	 RTLD_BOOTSTRAP) because rtld.c contains the common defn for
	 _dl_rtld_map, which is incompatible with a weak decl in the same
	 file.  */
# ifndef SHARED
      weak_extern (GL(dl_rtld_map));
# endif
      if (map != &GL(dl_rtld_map)) /* Already done in rtld itself.  */
# if !defined DO_RELA || defined ELF_MACHINE_REL_RELATIVE
	/* Rela platforms get the offset from r_addend and this must
	   be copied in the relocation address.  Therefore we can skip
	   the relative relocations only if this is for rel
	   relocations or rela relocations if they are computed as
	   memory_loc += l_addr...  */
	if (l_addr != 0)
# else
	/* ...or we know the object has been prelinked.  */
	if (l_addr != 0 || ! map->l_info[VALIDX(DT_GNU_PRELINKED)])
# endif
#endif
	  for (; relative < r; ++relative)
	    DO_ELF_MACHINE_REL_RELATIVE (map, l_addr, relative);

#ifdef RTLD_BOOTSTRAP
      /* The dynamic linker always uses versioning.  */
      assert (map->l_info[VERSYMIDX (DT_VERSYM)] != NULL);
#else
      if (map->l_info[VERSYMIDX (DT_VERSYM)])
#endif
	{
	  const ElfW(Half) *const version =
	    (const void *) D_PTR (map, l_info[VERSYMIDX (DT_VERSYM)]);

	  for (; r < end; ++r)
	    {
#if defined ELF_MACHINE_IRELATIVE && !defined RTLD_BOOTSTRAP
	      if (ELFW(R_TYPE) (r->r_info) == ELF_MACHINE_IRELATIVE)
		{
		  if (r2 == NULL)
		    r2 = r;
		  end2 = r;
		  continue;
		}
#endif

	      ElfW(Half) ndx = version[ELFW(R_SYM) (r->r_info)] & 0x7fff;
	      elf_machine_rel (map, r, &symtab[ELFW(R_SYM) (r->r_info)],
			       &map->l_versions[ndx],
			       (void *) (l_addr + r->r_offset), skip_ifunc
#ifndef NESTING
			       , boot_map
#endif
			       );
	    }

#if defined ELF_MACHINE_IRELATIVE && !defined RTLD_BOOTSTRAP
	  if (r2 != NULL)
	    for (; r2 <= end2; ++r2)
	      if (ELFW(R_TYPE) (r2->r_info) == ELF_MACHINE_IRELATIVE)
		{
		  ElfW(Half) ndx
		    = version[ELFW(R_SYM) (r2->r_info)] & 0x7fff;
		  elf_machine_rel (map, r2,
				   &symtab[ELFW(R_SYM) (r2->r_info)],
				   &map->l_versions[ndx],
				   (void *) (l_addr + r2->r_offset),
				   skip_ifunc
#ifndef NESTING
				   , boot_map
#endif
				   );
		}
#endif
	}
#ifndef RTLD_BOOTSTRAP
      else
	{
	  for (; r < end; ++r)
# ifdef ELF_MACHINE_IRELATIVE
	    if (ELFW(R_TYPE) (r->r_info) == ELF_MACHINE_IRELATIVE)
	      {
		if (r2 == NULL)
		  r2 = r;
		end2 = r;
	      }
	    else
# endif
	      elf_machine_rel (map, r, &symtab[ELFW(R_SYM) (r->r_info)], NULL,
			       (void *) (l_addr + r->r_offset), skip_ifunc
#ifndef NESTING
			       , boot_map
#endif
			       );

# ifdef ELF_MACHINE_IRELATIVE
	  if (r2 != NULL)
	    for (; r2 <= end2; ++r2)
	      if (ELFW(R_TYPE) (r2->r_info) == ELF_MACHINE_IRELATIVE)
		elf_machine_rel (map, r2, &symtab[ELFW(R_SYM) (r2->r_info)],
				 NULL, (void *) (l_addr + r2->r_offset),
				 skip_ifunc
#ifndef NESTING
				 , boot_map
#endif
				 );
# endif
	}
#endif
    }
}

#undef elf_dynamic_do_Rel
#undef Rel
#undef elf_machine_rel
#undef elf_machine_rel_relative
#undef DO_ELF_MACHINE_REL_RELATIVE
#undef DO_RELA
