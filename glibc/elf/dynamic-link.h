/* Inline functions for dynamic linking.
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

#ifndef NESTING
#define auto static
#endif

/* This macro is used as a callback from elf_machine_rel{a,} when a
   static TLS reloc is about to be performed.  Since (in dl-load.c) we
   permit dynamic loading of objects that might use such relocs, we
   have to check whether each use is actually doable.  If the object
   whose TLS segment the reference resolves to was allocated space in
   the static TLS block at startup, then it's ok.  Otherwise, we make
   an attempt to allocate it in surplus space on the fly.  If that
   can't be done, we fall back to the error that DF_STATIC_TLS is
   intended to produce.  */
#define HAVE_STATIC_TLS(map, sym_map)					\
    (__builtin_expect ((sym_map)->l_tls_offset != NO_TLS_OFFSET		\
		       && ((sym_map)->l_tls_offset			\
			   != FORCED_DYNAMIC_TLS_OFFSET), 1))

#define CHECK_STATIC_TLS(map, sym_map)					\
    do {								\
      if (!HAVE_STATIC_TLS (map, sym_map))				\
	_dl_allocate_static_tls (sym_map);				\
    } while (0)

#define TRY_STATIC_TLS(map, sym_map)					\
    (__builtin_expect ((sym_map)->l_tls_offset				\
		       != FORCED_DYNAMIC_TLS_OFFSET, 1)			\
     && (__builtin_expect ((sym_map)->l_tls_offset != NO_TLS_OFFSET, 1)	\
	 || _dl_try_allocate_static_tls (sym_map, true) == 0))

int _dl_try_allocate_static_tls (struct link_map *map, bool optional)
  attribute_hidden;

#include <elf.h>

#ifdef RESOLVE_MAP
/* We pass reloc_addr as a pointer to void, as opposed to a pointer to
   ElfW(Addr), because not all architectures can assume that the
   relocated address is properly aligned, whereas the compiler is
   entitled to assume that a pointer to a type is properly aligned for
   the type.  Even if we cast the pointer back to some other type with
   less strict alignment requirements, the compiler might still
   remember that the pointer was originally more aligned, thereby
   optimizing away alignment tests or using word instructions for
   copying memory, breaking the very code written to handle the
   unaligned cases.  */
# if ! ELF_MACHINE_NO_REL
auto inline void __attribute__((always_inline))
elf_machine_rel (struct link_map *map, const ElfW(Rel) *reloc,
		 const ElfW(Sym) *sym, const struct r_found_version *version,
		 void *const reloc_addr, int skip_ifunc);
auto inline void __attribute__((always_inline))
elf_machine_rel_relative (ElfW(Addr) l_addr, const ElfW(Rel) *reloc,
			  void *const reloc_addr);
# endif
# if ! ELF_MACHINE_NO_RELA
auto inline void __attribute__((always_inline))
elf_machine_rela (struct link_map *map, const ElfW(Rela) *reloc,
		  const ElfW(Sym) *sym, const struct r_found_version *version,
		  void *const reloc_addr, int skip_ifunc
#ifndef NESTING
		  , struct link_map *boot_map
#endif
		  );
auto inline void __attribute__((always_inline))
elf_machine_rela_relative (ElfW(Addr) l_addr, const ElfW(Rela) *reloc,
			   void *const reloc_addr);
# endif
# if ELF_MACHINE_NO_RELA || defined ELF_MACHINE_PLT_REL
auto inline void __attribute__((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      ElfW(Addr) l_addr, const ElfW(Rel) *reloc,
		      int skip_ifunc);
# else
auto inline void __attribute__((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      ElfW(Addr) l_addr, const ElfW(Rela) *reloc,
		      int skip_ifunc);
# endif
#endif

#include <dl-machine.h>

#include "get-dynamic-info.h"

#ifdef RESOLVE_MAP

# if defined RTLD_BOOTSTRAP || defined STATIC_PIE_BOOTSTRAP
#  define ELF_DURING_STARTUP (1)
# else
#  define ELF_DURING_STARTUP (0)
# endif

/* Get the definitions of `elf_dynamic_do_rel' and `elf_dynamic_do_rela'.
   These functions are almost identical, so we use cpp magic to avoid
   duplicating their code.  It cannot be done in a more general function
   because we must be able to completely inline.  */

/* On some machines, notably SPARC, DT_REL* includes DT_JMPREL in its
   range.  Note that according to the ELF spec, this is completely legal!

   We are guarenteed that we have one of three situations.  Either DT_JMPREL
   comes immediately after DT_REL*, or there is overlap and DT_JMPREL
   consumes precisely the very end of the DT_REL*, or DT_JMPREL and DT_REL*
   are completely separate and there is a gap between them.  */

#ifndef NESTING
# define _ELF_DYNAMIC_DO_RELOC(RELOC, reloc, map, do_lazy, skip_ifunc, test_rel, boot_map) \
  do {									      \
    struct { ElfW(Addr) start, size;					      \
	     __typeof (((ElfW(Dyn) *) 0)->d_un.d_val) nrelative; int lazy; }  \
      ranges[2] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };			      \
									      \
    if ((map)->l_info[DT_##RELOC])					      \
      {									      \
	ranges[0].start = D_PTR ((map), l_info[DT_##RELOC]);		      \
	ranges[0].size = (map)->l_info[DT_##RELOC##SZ]->d_un.d_val;	      \
	if (map->l_info[VERSYMIDX (DT_##RELOC##COUNT)] != NULL)		      \
	  ranges[0].nrelative						      \
	    = map->l_info[VERSYMIDX (DT_##RELOC##COUNT)]->d_un.d_val;	      \
      }									      \
    if ((map)->l_info[DT_PLTREL]					      \
	&& (!test_rel || (map)->l_info[DT_PLTREL]->d_un.d_val == DT_##RELOC)) \
      {									      \
	ElfW(Addr) start = D_PTR ((map), l_info[DT_JMPREL]);		      \
	ElfW(Addr) size = (map)->l_info[DT_PLTRELSZ]->d_un.d_val;	      \
									      \
	if (ranges[0].start + ranges[0].size == (start + size))		      \
	  ranges[0].size -= size;					      \
	if (ELF_DURING_STARTUP						      \
	    || (!(do_lazy)						      \
		&& (ranges[0].start + ranges[0].size) == start))	      \
	  {								      \
	    /* Combine processing the sections.  */			      \
	    ranges[0].size += size;					      \
	  }								      \
	else								      \
	  {								      \
	    ranges[1].start = start;					      \
	    ranges[1].size = size;					      \
	    ranges[1].lazy = (do_lazy);					      \
	  }								      \
      }									      \
									      \
    if (ELF_DURING_STARTUP)						      \
      elf_dynamic_do_##reloc ((map), ranges[0].start, ranges[0].size,	      \
			      ranges[0].nrelative, 0, skip_ifunc, boot_map);	\
    else								      \
      {									      \
	int ranges_index;						      \
	for (ranges_index = 0; ranges_index < 2; ++ranges_index)	      \
	  elf_dynamic_do_##reloc ((map),				      \
				  ranges[ranges_index].start,		      \
				  ranges[ranges_index].size,		      \
				  ranges[ranges_index].nrelative,	      \
				  ranges[ranges_index].lazy,		      \
				  skip_ifunc, boot_map);				\
      }									      \
  } while (0)
#else /* NESTING */
# define _ELF_DYNAMIC_DO_RELOC(RELOC, reloc, map, do_lazy, skip_ifunc, test_rel) \
  do {									      \
    struct { ElfW(Addr) start, size;					      \
	     __typeof (((ElfW(Dyn) *) 0)->d_un.d_val) nrelative; int lazy; }  \
      ranges[2] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };			      \
									      \
    if ((map)->l_info[DT_##RELOC])					      \
      {									      \
	ranges[0].start = D_PTR ((map), l_info[DT_##RELOC]);		      \
	ranges[0].size = (map)->l_info[DT_##RELOC##SZ]->d_un.d_val;	      \
	if (map->l_info[VERSYMIDX (DT_##RELOC##COUNT)] != NULL)		      \
	  ranges[0].nrelative						      \
	    = map->l_info[VERSYMIDX (DT_##RELOC##COUNT)]->d_un.d_val;	      \
      }									      \
    if ((map)->l_info[DT_PLTREL]					      \
	&& (!test_rel || (map)->l_info[DT_PLTREL]->d_un.d_val == DT_##RELOC)) \
      {									      \
	ElfW(Addr) start = D_PTR ((map), l_info[DT_JMPREL]);		      \
	ElfW(Addr) size = (map)->l_info[DT_PLTRELSZ]->d_un.d_val;	      \
									      \
	if (ranges[0].start + ranges[0].size == (start + size))		      \
	  ranges[0].size -= size;					      \
	if (ELF_DURING_STARTUP						      \
	    || (!(do_lazy)						      \
		&& (ranges[0].start + ranges[0].size) == start))	      \
	  {								      \
	    /* Combine processing the sections.  */			      \
	    ranges[0].size += size;					      \
	  }								      \
	else								      \
	  {								      \
	    ranges[1].start = start;					      \
	    ranges[1].size = size;					      \
	    ranges[1].lazy = (do_lazy);					      \
	  }								      \
      }									      \
									      \
    if (ELF_DURING_STARTUP)						      \
      elf_dynamic_do_##reloc ((map), ranges[0].start, ranges[0].size,	      \
			      ranges[0].nrelative, 0, skip_ifunc);	      \
    else								      \
      {									      \
	int ranges_index;						      \
	for (ranges_index = 0; ranges_index < 2; ++ranges_index)	      \
	  elf_dynamic_do_##reloc ((map),				      \
				  ranges[ranges_index].start,		      \
				  ranges[ranges_index].size,		      \
				  ranges[ranges_index].nrelative,	      \
				  ranges[ranges_index].lazy,		      \
				  skip_ifunc);				      \
      }									      \
  } while (0)
#endif /* NESTING */

# if ELF_MACHINE_NO_REL || ELF_MACHINE_NO_RELA
#  define _ELF_CHECK_REL 0
# else
#  define _ELF_CHECK_REL 1
# endif

#ifndef NESTING
# if ! ELF_MACHINE_NO_REL
#  include "do-rel.h"
#  define ELF_DYNAMIC_DO_REL(map, lazy, skip_ifunc, boot_map)			\
  _ELF_DYNAMIC_DO_RELOC (REL, Rel, map, lazy, skip_ifunc, _ELF_CHECK_REL, boot_map)
# else
#  define ELF_DYNAMIC_DO_REL(map, lazy, skip_ifunc, boot_map) /* Nothing to do.  */
# endif

# if ! ELF_MACHINE_NO_RELA
#  define DO_RELA
#  include "do-rel.h"
#  define ELF_DYNAMIC_DO_RELA(map, lazy, skip_ifunc, boot_map)			\
  _ELF_DYNAMIC_DO_RELOC (RELA, Rela, map, lazy, skip_ifunc, _ELF_CHECK_REL, boot_map)
# else
#  define ELF_DYNAMIC_DO_RELA(map, lazy, skip_ifunc, boot_map) /* Nothing to do.  */
# endif

/* This can't just be an inline function because GCC is too dumb
   to inline functions containing inlines themselves.  */
# define ELF_DYNAMIC_RELOCATE(map, lazy, consider_profile, skip_ifunc, boot_map) \
  do {									      \
    int edr_lazy = elf_machine_runtime_setup ((map), (lazy),		      \
					      (consider_profile));	      \
    ELF_DYNAMIC_DO_REL ((map), edr_lazy, skip_ifunc, boot_map);			\
    ELF_DYNAMIC_DO_RELA ((map), edr_lazy, skip_ifunc, boot_map);			\
  } while (0)
#else /* NESTING */
# if ! ELF_MACHINE_NO_REL
#  include "do-rel.h"
#  define ELF_DYNAMIC_DO_REL(map, lazy, skip_ifunc) \
  _ELF_DYNAMIC_DO_RELOC (REL, Rel, map, lazy, skip_ifunc, _ELF_CHECK_REL)
# else
#  define ELF_DYNAMIC_DO_REL(map, lazy, skip_ifunc) /* Nothing to do.  */
# endif

# if ! ELF_MACHINE_NO_RELA
#  define DO_RELA
#  include "do-rel.h"
#  define ELF_DYNAMIC_DO_RELA(map, lazy, skip_ifunc) \
  _ELF_DYNAMIC_DO_RELOC (RELA, Rela, map, lazy, skip_ifunc, _ELF_CHECK_REL)
# else
#  define ELF_DYNAMIC_DO_RELA(map, lazy, skip_ifunc) /* Nothing to do.  */
# endif

/* This can't just be an inline function because GCC is too dumb
   to inline functions containing inlines themselves.  */
# define ELF_DYNAMIC_RELOCATE(map, lazy, consider_profile, skip_ifunc) \
  do {									      \
    int edr_lazy = elf_machine_runtime_setup ((map), (lazy),		      \
					      (consider_profile));	      \
    ELF_DYNAMIC_DO_REL ((map), edr_lazy, skip_ifunc);			      \
    ELF_DYNAMIC_DO_RELA ((map), edr_lazy, skip_ifunc);			      \
  } while (0)
#endif /* NESTING */

#endif
