/* Copyright (C) 1995-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef dl_machine_h
#define dl_machine_h

#define ELF_MACHINE_NAME "microblaze"

#include <sys/param.h>
#include <tls.h>

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return (ehdr->e_machine == EM_MICROBLAZE);
}

/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  This must be inlined in a function which
   uses global data.  */
static inline Elf32_Addr
elf_machine_dynamic (void)
{
  /* This produces a GOTOFF reloc that resolves to zero at link time, so in
     fact just loads from the GOT register directly.  By doing it without
     an asm we can let the compiler choose any register.  */

  Elf32_Addr got_entry_0;
  __asm__ __volatile__(
    "lwi %0,r20,0"
    :"=r"(got_entry_0)
    );
  return got_entry_0;
}

/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr
elf_machine_load_address (void)
{
  /* Compute the difference between the runtime address of _DYNAMIC as seen
     by a GOTOFF reference, and the link-time address found in the special
     unrelocated first GOT entry.  */

  Elf32_Addr dyn;
  __asm__ __volatile__ (
    "addik %0,r20,_DYNAMIC@GOTOFF"
    : "=r"(dyn)
    );
  return dyn - elf_machine_dynamic ();
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((always_inline))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  extern void _dl_runtime_resolve (Elf32_Word);
  extern void _dl_runtime_profile (Elf32_Word);

  return lazy;
}

/* The PLT uses Elf32_Rela relocs.  */
#define elf_machine_relplt elf_machine_rela

/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK	0x80000000UL

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_START asm ("\
	.text\n\
	.globl _start\n\
	.type _start,@function\n\
_start:\n\
	addk  r5,r0,r1\n\
	addk  r3,r0,r0\n\
1:\n\
	addik r5,r5,4\n\
	lw    r4,r5,r0\n\
	bneid r4,1b\n\
	addik r3,r3,1\n\
	addik r3,r3,-1\n\
	addk  r5,r0,r1\n\
	sw    r3,r5,r0\n\
	addik r1,r1,-24\n\
	sw    r15,r1,r0\n\
	brlid r15,_dl_start\n\
	nop\n\
	/* FALLTHRU.  */\n\
\n\
	.globl _dl_start_user\n\
	.type _dl_start_user,@function\n\
_dl_start_user:\n\
	mfs   r20,rpc\n\
	addik r20,r20,_GLOBAL_OFFSET_TABLE_+8\n\
	lwi   r4,r20,_dl_skip_args@GOTOFF\n\
	lwi   r5,r1,24\n\
	rsubk r5,r4,r5\n\
	addk  r4,r4,r4\n\
	addk  r4,r4,r4\n\
	addk  r1,r1,r4\n\
	swi   r5,r1,24\n\
	swi   r3,r1,20\n\
	addk  r6,r5,r0\n\
	addk  r5,r5,r5\n\
	addk  r5,r5,r5\n\
	addik r7,r1,28\n\
	addk  r8,r7,r5\n\
	addik r8,r8,4\n\
	lwi   r5,r20,_rtld_local@GOTOFF\n\
	brlid r15,_dl_init\n\
	nop\n\
	lwi   r5,r1,24\n\
	lwi   r3,r1,20\n\
	addk  r4,r5,r5\n\
	addk  r4,r4,r4\n\
	addik r6,r1,28\n\
	addk  r7,r6,r4\n\
	addik r7,r7,4\n\
	addik r15,r20,_dl_fini@GOTOFF\n\
	addik r15,r15,-8\n\
	brad  r3\n\
	addik r1,r1,24\n\
	nop\n\
	.size _dl_start_user, . - _dl_start_user\n\
	.previous");

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry or
   TLS variable, so undefined references should not be allowed to
   define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#ifndef RTLD_BOOTSTRAP
# define elf_machine_type_class(type) \
  (((type) == R_MICROBLAZE_JUMP_SLOT \
    || (type) == R_MICROBLAZE_TLSDTPREL32 \
    || (type) == R_MICROBLAZE_TLSDTPMOD32 \
    || (type) == R_MICROBLAZE_TLSTPREL32) \
    * ELF_RTYPE_CLASS_PLT \
   | ((type) == R_MICROBLAZE_COPY) * ELF_RTYPE_CLASS_COPY)
#else
# define elf_machine_type_class(type) \
  (((type) == R_MICROBLAZE_JUMP_SLOT) * ELF_RTYPE_CLASS_PLT \
   | ((type) == R_MICROBLAZE_COPY) * ELF_RTYPE_CLASS_COPY)
#endif

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	R_MICROBLAZE_JUMP_SLOT

/* The microblaze never uses Elf32_Rel relocations.  */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

static inline Elf32_Addr
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf32_Rela *reloc,
		       Elf32_Addr *reloc_addr, Elf32_Addr value)
{
  return *reloc_addr = value;
}

/* Return the final value of a plt relocation. Ignore the addend.  */
static inline Elf32_Addr
elf_machine_plt_value (struct link_map *map, const Elf32_Rela *reloc,
		       Elf32_Addr value)
{
  return value;
}

#endif /* !dl_machine_h.  */

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER microblaze_gnu_pltenter
#define ARCH_LA_PLTEXIT microblaze_gnu_pltexit

#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

/* Macro to put 32-bit relocation value into 2 words.  */
#define PUT_REL_64(rel_addr,val) \
  do { \
    ((unsigned short *)(rel_addr))[1] = (val) >> 16; \
    ((unsigned short *)(rel_addr))[3] = (val) & 0xffff; \
  } while (0)

auto inline void __attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const Elf32_Rela *reloc,
		  const Elf32_Sym *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const int r_type = ELF32_R_TYPE (reloc->r_info);

  if (__builtin_expect (r_type == R_MICROBLAZE_64_PCREL, 0))
    PUT_REL_64 (reloc_addr, map->l_addr + reloc->r_addend);
  else if (r_type == R_MICROBLAZE_REL)
    *reloc_addr = map->l_addr + reloc->r_addend;
  else
    {
      const Elf32_Sym *const refsym = sym;
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf32_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      value += reloc->r_addend;
      if (r_type == R_MICROBLAZE_GLOB_DAT
          || r_type == R_MICROBLAZE_JUMP_SLOT
          || r_type == R_MICROBLAZE_32)
	{
	  *reloc_addr = value;
	}
      else if (r_type == R_MICROBLAZE_COPY)
	{
	  if (sym != NULL && (sym->st_size > refsym->st_size
	      || (sym->st_size < refsym->st_size && GLRO (dl_verbose))) )
	    {
	      const char *strtab;

	      strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
	      _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				RTLD_PROGNAME, strtab + refsym->st_name);
	    }
	  memcpy (reloc_addr_arg, (void *) value,
		  MIN (sym->st_size, refsym->st_size));
	}
      else if (r_type == R_MICROBLAZE_NONE)
	{
	}
#if !defined RTLD_BOOTSTRAP
      else if (r_type == R_MICROBLAZE_TLSDTPMOD32)
	{
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
	}
      else if (r_type == R_MICROBLAZE_TLSDTPREL32)
	{
	  if (sym != NULL)
	    *reloc_addr = sym->st_value + reloc->r_addend;
	}
      else if (r_type == R_MICROBLAZE_TLSTPREL32)
	{
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = sym->st_value + sym_map->l_tls_offset + reloc->r_addend;
	    }
	}
#endif
      else
	{
	  _dl_reloc_bad_type (map, r_type, 0);
	}
    }
}

auto inline void
elf_machine_rela_relative (Elf32_Addr l_addr, const Elf32_Rela *reloc,
			   void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  PUT_REL_64 (reloc_addr, l_addr + reloc->r_addend);
}

auto inline void
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rela *reloc,
		      int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  if (ELF32_R_TYPE (reloc->r_info) == R_MICROBLAZE_JUMP_SLOT)
    *reloc_addr += l_addr;
  else
    _dl_reloc_bad_type (map, ELF32_R_TYPE (reloc->r_info), 1);
}

#endif /* RESOLVE_MAP.  */
