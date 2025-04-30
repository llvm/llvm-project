/* Machine-dependent ELF dynamic relocation inline functions.  PowerPC version.
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

#ifndef dl_machine_h
#define dl_machine_h

#define ELF_MACHINE_NAME "powerpc"

#include <assert.h>
#include <dl-tls.h>
#include <dl-irel.h>
#include <hwcapinfo.h>

/* Translate a processor specific dynamic tag to the index
   in l_info array.  */
#define DT_PPC(x) (DT_PPC_##x - DT_LOPROC + DT_NUM)

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_PPC;
}

/* Return the value of the GOT pointer.  */
static inline Elf32_Addr * __attribute__ ((const))
ppc_got (void)
{
  Elf32_Addr *got;

  asm ("bcl 20,31,1f\n"
       "1:	mflr %0\n"
       "	addis %0,%0,_GLOBAL_OFFSET_TABLE_-1b@ha\n"
       "	addi %0,%0,_GLOBAL_OFFSET_TABLE_-1b@l\n"
       : "=b" (got) : : "lr");

  return got;
}

/* Return the link-time address of _DYNAMIC, stored as
   the first value in the GOT. */
static inline Elf32_Addr __attribute__ ((const))
elf_machine_dynamic (void)
{
  return *ppc_got ();
}

/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr __attribute__ ((const))
elf_machine_load_address (void)
{
  Elf32_Addr *branchaddr;
  Elf32_Addr runtime_dynamic;

  /* This is much harder than you'd expect.  Possibly I'm missing something.
     The 'obvious' way:

       Apparently, "bcl 20,31,$+4" is what should be used to load LR
       with the address of the next instruction.
       I think this is so that machines that do bl/blr pairing don't
       get confused.

     asm ("bcl 20,31,0f ;"
	  "0: mflr 0 ;"
	  "lis %0,0b@ha;"
	  "addi %0,%0,0b@l;"
	  "subf %0,%0,0"
	  : "=b" (addr) : : "r0", "lr");

     doesn't work, because the linker doesn't have to (and in fact doesn't)
     update the @ha and @l references; the loader (which runs after this
     code) will do that.

     Instead, we use the following trick:

     The linker puts the _link-time_ address of _DYNAMIC at the first
     word in the GOT. We could branch to that address, if we wanted,
     by using an @local reloc; the linker works this out, so it's safe
     to use now. We can't, of course, actually branch there, because
     we'd cause an illegal instruction exception; so we need to compute
     the address ourselves. That gives us the following code: */

  /* Get address of the 'b _DYNAMIC@local'...  */
  asm ("bcl 20,31,0f;"
       "b _DYNAMIC@local;"
       "0:"
       : "=l" (branchaddr));

  /* So now work out the difference between where the branch actually points,
     and the offset of that location in memory from the start of the file.  */
  runtime_dynamic = ((Elf32_Addr) branchaddr
		     + ((Elf32_Sword) (*branchaddr << 6 & 0xffffff00) >> 6));

  return runtime_dynamic - elf_machine_dynamic ();
}

#define ELF_MACHINE_BEFORE_RTLD_RELOC(dynamic_info) /* nothing */

/* The PLT uses Elf32_Rela relocs.  */
#define elf_machine_relplt elf_machine_rela

/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK	0xf0000000UL

/* The actual _start code is in dl-start.S.  Use a really
   ugly bit of assembler to let dl-start.o see _dl_start.  */
#define RTLD_START asm (".globl _dl_start");

/* Decide where a relocatable object should be loaded.  */
extern ElfW(Addr)
__elf_preferred_address(struct link_map *loader, size_t maplength,
			ElfW(Addr) mapstartpref);
#define ELF_PREFERRED_ADDRESS(loader, maplength, mapstartpref) \
  __elf_preferred_address (loader, maplength, mapstartpref)

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry, so
   PLT entries should not be allowed to define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
/* We never want to use a PLT entry as the destination of a
   reloc, when what is being relocated is a branch. This is
   partly for efficiency, but mostly so we avoid loops.  */
#define elf_machine_type_class(type)			\
  ((((type) == R_PPC_JMP_SLOT				\
    || (type) == R_PPC_REL24				\
    || ((type) >= R_PPC_DTPMOD32 /* contiguous TLS */	\
	&& (type) <= R_PPC_DTPREL32)			\
    || (type) == R_PPC_ADDR24) * ELF_RTYPE_CLASS_PLT)	\
   | (((type) == R_PPC_COPY) * ELF_RTYPE_CLASS_COPY))

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	R_PPC_JMP_SLOT

/* The PowerPC never uses REL relocations.  */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

/* We define an initialization function to initialize HWCAP/HWCAP2 and
   platform data so it can be copied into the TCB later.  This is called
   very early in _dl_sysdep_start for dynamically linked binaries.  */
#ifdef SHARED
# define DL_PLATFORM_INIT dl_platform_init ()

static inline void __attribute__ ((unused))
dl_platform_init (void)
{
  __tcb_parse_hwcap_and_convert_at_platform ();
}
#endif

/* Set up the loaded object described by MAP so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.
   Also install a small trampoline to be used by entries that have
   been relocated to an address too far away for a single branch.  */
extern int __elf_machine_runtime_setup (struct link_map *map,
					int lazy, int profile);

static inline int
elf_machine_runtime_setup (struct link_map *map,
			   int lazy, int profile)
{
  if (map->l_info[DT_JMPREL] == 0)
    return lazy;

  if (map->l_info[DT_PPC(GOT)] == 0)
    /* Handle old style PLT.  */
    return __elf_machine_runtime_setup (map, lazy, profile);

  /* New style non-exec PLT consisting of an array of addresses.  */
  map->l_info[DT_PPC(GOT)]->d_un.d_ptr += map->l_addr;
  if (lazy)
    {
      Elf32_Addr *plt, *got, glink;
      Elf32_Word num_plt_entries;
      void (*dlrr) (void);
      extern void _dl_runtime_resolve (void);
      extern void _dl_prof_resolve (void);

      if (__glibc_likely (!profile))
	dlrr = _dl_runtime_resolve;
      else
	{
	  if (GLRO(dl_profile) != NULL
	      &&_dl_name_match_p (GLRO(dl_profile), map))
	    GL(dl_profile_map) = map;
	  dlrr = _dl_prof_resolve;
	}
      got = (Elf32_Addr *) map->l_info[DT_PPC(GOT)]->d_un.d_ptr;
      glink = got[1];
      got[1] = (Elf32_Addr) dlrr;
      got[2] = (Elf32_Addr) map;

      /* Relocate everything in .plt by the load address offset.  */
      plt = (Elf32_Addr *) D_PTR (map, l_info[DT_PLTGOT]);
      num_plt_entries = (map->l_info[DT_PLTRELSZ]->d_un.d_val
			 / sizeof (Elf32_Rela));

      /* If a library is prelinked but we have to relocate anyway,
	 we have to be able to undo the prelinking of .plt section.
	 The prelinker saved us at got[1] address of .glink
	 section's start.  */
      if (glink)
	{
	  glink += map->l_addr;
	  while (num_plt_entries-- != 0)
	    *plt++ = glink, glink += 4;
	}
      else
	while (num_plt_entries-- != 0)
	  *plt++ += map->l_addr;
    }
  return lazy;
}

/* Change the PLT entry whose reloc is 'reloc' to call the actual routine.  */
extern Elf32_Addr __elf_machine_fixup_plt (struct link_map *map,
					   Elf32_Addr *reloc_addr,
					   Elf32_Addr finaladdr);

static inline Elf32_Addr
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf32_Rela *reloc,
		       Elf32_Addr *reloc_addr, Elf64_Addr finaladdr)
{
  if (map->l_info[DT_PPC(GOT)] == 0)
    /* Handle old style PLT.  */
    return __elf_machine_fixup_plt (map, reloc_addr, finaladdr);

  *reloc_addr = finaladdr;
  return finaladdr;
}

/* Return the final value of a plt relocation.  */
static inline Elf32_Addr
elf_machine_plt_value (struct link_map *map, const Elf32_Rela *reloc,
		       Elf32_Addr value)
{
  return value + reloc->r_addend;
}


/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER ppc32_gnu_pltenter
#define ARCH_LA_PLTEXIT ppc32_gnu_pltexit

#endif /* dl_machine_h */

#ifdef RESOLVE_MAP

/* Do the actual processing of a reloc, once its target address
   has been determined.  */
extern void __process_machine_rela (struct link_map *map,
				    const Elf32_Rela *reloc,
				    struct link_map *sym_map,
				    const Elf32_Sym *sym,
				    const Elf32_Sym *refsym,
				    Elf32_Addr *const reloc_addr,
				    Elf32_Addr finaladdr,
				    int rinfo, bool skip_ifunc)
  attribute_hidden;

/* Call _dl_signal_error when a resolved value overflows a relocated area.  */
extern void _dl_reloc_overflow (struct link_map *map,
				const char *name,
				Elf32_Addr *const reloc_addr,
				const Elf32_Sym *refsym) attribute_hidden;

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   LOADADDR is the load address of the object; INFO is an array indexed
   by DT_* of the .dynamic section info.  */

auto inline void __attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const Elf32_Rela *reloc,
		  const Elf32_Sym *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const Elf32_Sym *const refsym = sym;
  Elf32_Addr value;
  const int r_type = ELF32_R_TYPE (reloc->r_info);
  struct link_map *sym_map = NULL;

#ifndef RESOLVE_CONFLICT_FIND_MAP
  if (r_type == R_PPC_RELATIVE)
    {
      *reloc_addr = map->l_addr + reloc->r_addend;
      return;
    }

  if (__glibc_unlikely (r_type == R_PPC_NONE))
    return;

  /* binutils on ppc32 includes st_value in r_addend for relocations
     against local symbols.  */
  if (__builtin_expect (ELF32_ST_BIND (sym->st_info) == STB_LOCAL, 0)
      && sym->st_shndx != SHN_UNDEF)
    {
      sym_map = map;
      value = map->l_addr;
    }
  else
    {
      sym_map = RESOLVE_MAP (&sym, version, r_type);
      value = SYMBOL_ADDRESS (sym_map, sym, true);
    }
  value += reloc->r_addend;
#else
  value = reloc->r_addend;
#endif

  if (sym != NULL
      && __builtin_expect (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC, 0)
      && __builtin_expect (sym->st_shndx != SHN_UNDEF, 1)
      && __builtin_expect (!skip_ifunc, 1))
    value = elf_ifunc_invoke (value);

  /* A small amount of code is duplicated here for speed.  In libc,
     more than 90% of the relocs are R_PPC_RELATIVE; in the X11 shared
     libraries, 60% are R_PPC_RELATIVE, 24% are R_PPC_GLOB_DAT or
     R_PPC_ADDR32, and 16% are R_PPC_JMP_SLOT (which this routine
     wouldn't usually handle).  As an bonus, doing this here allows
     the switch statement in __process_machine_rela to work.  */
  switch (r_type)
    {
    case R_PPC_GLOB_DAT:
    case R_PPC_ADDR32:
      *reloc_addr = value;
      break;

#ifndef RESOLVE_CONFLICT_FIND_MAP
# ifdef RTLD_BOOTSTRAP
#  define NOT_BOOTSTRAP 0
# else
#  define NOT_BOOTSTRAP 1
# endif

    case R_PPC_DTPMOD32:
      if (map->l_info[DT_PPC(OPT)]
	  && (map->l_info[DT_PPC(OPT)]->d_un.d_val & PPC_OPT_TLS))
	{
	  if (!NOT_BOOTSTRAP)
	    {
	      reloc_addr[0] = 0;
	      reloc_addr[1] = (sym_map->l_tls_offset - TLS_TP_OFFSET
			       + TLS_DTV_OFFSET);
	      break;
	    }
	  else if (sym_map != NULL)
	    {
# ifndef SHARED
	      CHECK_STATIC_TLS (map, sym_map);
# else
	      if (TRY_STATIC_TLS (map, sym_map))
# endif
		{
		  reloc_addr[0] = 0;
		  /* Set up for local dynamic.  */
		  reloc_addr[1] = (sym_map->l_tls_offset - TLS_TP_OFFSET
				   + TLS_DTV_OFFSET);
		  break;
		}
	    }
	}
      if (!NOT_BOOTSTRAP)
	/* During startup the dynamic linker is always index 1.  */
	*reloc_addr = 1;
      else if (sym_map != NULL)
	/* Get the information from the link map returned by the
	   RESOLVE_MAP function.  */
	*reloc_addr = sym_map->l_tls_modid;
      break;
    case R_PPC_DTPREL32:
      if (map->l_info[DT_PPC(OPT)]
	  && (map->l_info[DT_PPC(OPT)]->d_un.d_val & PPC_OPT_TLS))
	{
	  if (!NOT_BOOTSTRAP)
	    {
	      *reloc_addr = TLS_TPREL_VALUE (sym_map, sym, reloc);
	      break;
	    }
	  else if (sym_map != NULL)
	    {
	      /* This reloc is always preceded by R_PPC_DTPMOD32.  */
# ifndef SHARED
	      assert (HAVE_STATIC_TLS (map, sym_map));
# else
	      if (HAVE_STATIC_TLS (map, sym_map))
# endif
		{
		  *reloc_addr = TLS_TPREL_VALUE (sym_map, sym, reloc);
		  break;
		}
	    }
	}
      /* During relocation all TLS symbols are defined and used.
	 Therefore the offset is already correct.  */
      if (NOT_BOOTSTRAP && sym_map != NULL)
	*reloc_addr = TLS_DTPREL_VALUE (sym, reloc);
      break;
    case R_PPC_TPREL32:
      if (!NOT_BOOTSTRAP || sym_map != NULL)
	{
	  if (NOT_BOOTSTRAP)
	    CHECK_STATIC_TLS (map, sym_map);
	  *reloc_addr = TLS_TPREL_VALUE (sym_map, sym, reloc);
	}
      break;
#endif

    case R_PPC_JMP_SLOT:
#ifdef RESOLVE_CONFLICT_FIND_MAP
      RESOLVE_CONFLICT_FIND_MAP (map, reloc_addr);
#endif
      if (map->l_info[DT_PPC(GOT)] != 0)
	{
	  *reloc_addr = value;
	  break;
	}
      /* FALLTHROUGH */

    default:
      __process_machine_rela (map, reloc, sym_map, sym, refsym,
			      reloc_addr, value, r_type, skip_ifunc);
    }
}

auto inline void __attribute__ ((always_inline))
elf_machine_rela_relative (Elf32_Addr l_addr, const Elf32_Rela *reloc,
			   void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  *reloc_addr = l_addr + reloc->r_addend;
}

auto inline void __attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rela *reloc,
		      int skip_ifunc)
{
  /* elf_machine_runtime_setup handles this. */
}

#endif /* RESOLVE_MAP */
