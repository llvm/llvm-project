/* Machine-dependent ELF dynamic relocation inline functions.  ARC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef dl_machine_h
#define dl_machine_h

#define ELF_MACHINE_NAME "arc"

#include <entry.h>

#ifndef ENTRY_POINT
# error ENTRY_POINT needs to be defined for ARC
#endif

#include <string.h>
#include <link.h>
#include <dl-tls.h>

/* Dynamic Linking ABI for ARCv2 ISA.

                        PLT
          --------------------------------	<---- DT_PLTGOT
          |  ld r11, [pcl, off-to-GOT[1] |  0
          |                              |  4
   plt0   |  ld r10, [pcl, off-to-GOT[2] |  8
          |                              | 12
          |  j [r10]                     | 16
          --------------------------------
          |    Base address of GOT       | 20
          --------------------------------
          |  ld r12, [pcl, off-to-GOT[3] | 24
   plt1   |                              |
          |  j.d    [r12]                | 32
          |  mov    r12, pcl             | 36
          --------------------------------
          |                              | 40
          ~                              ~
          ~                              ~
          |                              |
          --------------------------------

               .got
          --------------
          |    [0]     |
          |    ...     |  Runtime address for data symbols
          |    [n]     |
          --------------

            .got.plt
          --------------
          |    [0]     |  Build address of .dynamic
          --------------
          |    [1]     |  Module info - setup by ld.so
          --------------
          |    [2]     |  resolver entry point
          --------------
          |    [3]     |
          |    ...     |  Runtime address for function symbols
          |    [f]     |
          --------------

   For ARCompact, the PLT is 12 bytes due to short instructions

          --------------------------------
          |  ld r12, [pcl, off-to-GOT[3] | 24   (12 bytes each)
   plt1   |                              |
          |  j_s.d  [r12]                | 32
          |  mov_s  r12, pcl             | 34
          --------------------------------
          |                              | 36  */

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const ElfW(Ehdr) *ehdr)
{
  return (ehdr->e_machine == EM_ARCV2		 /* ARC HS.  */
	  || ehdr->e_machine == EM_ARC_COMPACT); /* ARC 700.  */
}

/* Get build time address of .dynamic as setup in GOT[0]
   This is called very early in _dl_start so it has not been relocated to
   runtime value.  */
static inline ElfW(Addr)
elf_machine_dynamic (void)
{
  extern const ElfW(Addr) _GLOBAL_OFFSET_TABLE_[] attribute_hidden;
  return _GLOBAL_OFFSET_TABLE_[0];
}


/* Return the run-time load address of the shared object.  */
static inline ElfW(Addr)
elf_machine_load_address (void)
{
  ElfW(Addr) build_addr, run_addr;

  /* For build address, below generates
     ld  r0, [pcl, _GLOBAL_OFFSET_TABLE_@pcl].  */
  build_addr = elf_machine_dynamic ();
  __asm__ ("add %0, pcl, _DYNAMIC@pcl	\n" : "=r" (run_addr));

  return run_addr - build_addr;
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int
__attribute__ ((always_inline))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  extern void _dl_runtime_resolve (void);

  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* On ARC DT_PLTGOT point to .plt whose 5th word (after the PLT header)
         contains the address of .got.  */
      ElfW(Addr) *plt_base = (ElfW(Addr) *) D_PTR (l, l_info[DT_PLTGOT]);
      ElfW(Addr) *got = (ElfW(Addr) *) (plt_base[5] + l->l_addr);

      got[1] = (ElfW(Addr)) l;	/* Identify this shared object.  */

      /* This function will get called to fix up the GOT entry indicated by
	 the offset on the stack, and then jump to the resolved address.  */
      got[2] = (ElfW(Addr)) &_dl_runtime_resolve;
    }

  return lazy;
}

/* What this code does:
    -ldso starts execution here when kernel returns from execve
    -calls into generic ldso entry point _dl_start
    -optionally adjusts argc for executable if exec passed as cmd
    -calls into app main with address of finaliser.  */

#define RTLD_START asm ("\
.text									\n\
.globl __start								\n\
.type __start, @function						\n\
__start:								\n\
	/* (1). bootstrap ld.so.  */					\n\
	bl.d    _dl_start                                       	\n\
	mov_s   r0, sp  /* pass ptr to aux vector tbl.    */    	\n\
	mov r13, r0	/* safekeep app elf entry point.  */		\n\
									\n\
	/* (2). If ldso ran with executable as arg.       */		\n\
	/*      skip the extra args calc by dl_start.     */		\n\
	ld_s    r1, [sp]       /* orig argc.  */			\n\
	ld      r12, [pcl, _dl_skip_args@pcl]                   	\n\
	breq	r12, 0, 1f						\n\
									\n\
	add2    sp, sp, r12 /* discard argv entries from stack.  */	\n\
	sub_s   r1, r1, r12 /* adjusted argc on stack.  */      	\n\
	st_s    r1, [sp]                                        	\n\
	add	r2, sp, 4						\n\
	/* intermediate LD for ST emcoding limitations.  */		\n\
	ld	r3, [pcl, _dl_argv@gotpc]    				\n\
	st	r2, [r3]						\n\
1:									\n\
	/* (3). call preinit stuff.  */					\n\
	ld	r0, [pcl, _rtld_local@pcl]				\n\
	add	r2, sp, 4	; argv					\n\
	add2	r3, r2, r1						\n\
	add	r3, r3, 4	; env					\n\
	bl	_dl_init@plt						\n\
									\n\
	/* (4) call app elf entry point.  */				\n\
	add     r0, pcl, _dl_fini@pcl					\n\
	j	[r13]							\n\
									\n\
	.size  __start,.-__start                               		\n\
	.previous                                               	\n\
");

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry, so
   PLT entries should not be allowed to define the value.
   ELF_RTYPE_CLASS_NOCOPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#define elf_machine_type_class(type)				\
  ((((type) == R_ARC_JUMP_SLOT					\
     || (type) == R_ARC_TLS_DTPMOD				\
     || (type) == R_ARC_TLS_DTPOFF				\
     || (type) == R_ARC_TLS_TPOFF) * ELF_RTYPE_CLASS_PLT)	\
   | (((type) == R_ARC_COPY) * ELF_RTYPE_CLASS_COPY))

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT  R_ARC_JUMP_SLOT

/* ARC uses Rela relocations.  */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

/* Fixup a PLT entry to bounce directly to the function at VALUE.  */

static inline ElfW(Addr)
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const ElfW(Rela) *reloc,
		       ElfW(Addr) *reloc_addr, ElfW(Addr) value)
{
  return *reloc_addr = value;
}

/* Return the final value of a plt relocation.  */
#define elf_machine_plt_value(map, reloc, value) (value)

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER arc_gnu_pltenter
#define ARCH_LA_PLTEXIT arc_gnu_pltexit

#endif /* dl_machine_h */

#ifdef RESOLVE_MAP

inline void
__attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const ElfW(Rela) *reloc,
                  const ElfW(Sym) *sym, const struct r_found_version *version,
                  void *const reloc_addr_arg, int skip_ifunc)
{
  ElfW(Addr) r_info = reloc->r_info;
  const unsigned long int r_type = ELFW (R_TYPE) (r_info);
  ElfW(Addr) *const reloc_addr = reloc_addr_arg;

  if (__glibc_unlikely (r_type == R_ARC_RELATIVE))
    *reloc_addr += map->l_addr;
  else if (__glibc_unlikely (r_type == R_ARC_NONE))
    return;
  else
    {
      const ElfW(Sym) *const refsym = sym;
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      ElfW(Addr) value = SYMBOL_ADDRESS (sym_map, sym, true);

      switch (r_type)
        {
        case R_ARC_COPY:
          if (__glibc_unlikely (sym == NULL))
            /* This can happen in trace mode if an object could not be
               found.  */
            break;

          size_t size = sym->st_size;
          if (__glibc_unlikely (size != refsym->st_size))
            {
              const char *strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
              if (sym->st_size > refsym->st_size)
                size = refsym->st_size;
              if (sym->st_size > refsym->st_size || GLRO(dl_verbose))
                _dl_error_printf ("\
  %s: Symbol `%s' has different size in shared object, consider re-linking\n",
                                  rtld_progname ?: "<program name unknown>",
                                  strtab + refsym->st_name);
            }

          memcpy (reloc_addr_arg, (void *) value, size);
          break;

        case R_ARC_GLOB_DAT:
        case R_ARC_JUMP_SLOT:
            *reloc_addr = value;
          break;

        case R_ARC_TLS_DTPMOD:
          if (sym_map != NULL)
            /* Get the information from the link map returned by the
               resolv function.  */
            *reloc_addr = sym_map->l_tls_modid;
          break;

        case R_ARC_TLS_DTPOFF:
          if (sym != NULL)
            /* Offset set by the linker in the GOT entry would be overwritten
               by dynamic loader instead of added to the symbol location.
               Other target have the same approach on DTPOFF relocs.  */
            *reloc_addr += sym->st_value;
          break;

        case R_ARC_TLS_TPOFF:
          if (sym != NULL)
            {
              CHECK_STATIC_TLS (map, sym_map);
              *reloc_addr = sym_map->l_tls_offset + sym->st_value + reloc->r_addend;
            }
          break;

        case R_ARC_32:
          *reloc_addr += value + reloc->r_addend;
          break;

        case R_ARC_PC32:
          *reloc_addr += value + reloc->r_addend - (unsigned long int) reloc_addr;
          break;

        default:
          _dl_reloc_bad_type (map, r_type, 0);
          break;
        }
    }
}

inline void
__attribute__ ((always_inline))
elf_machine_rela_relative (ElfW(Addr) l_addr, const ElfW(Rela) *reloc,
                           void *const reloc_addr_arg)
{
  ElfW(Addr) *const reloc_addr = reloc_addr_arg;
  *reloc_addr += l_addr;
}

inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map, ElfW(Addr) l_addr,
                      const ElfW(Rela) *reloc, int skip_ifunc)
{
  ElfW(Addr) *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELFW (R_TYPE) (reloc->r_info);

  if (r_type == R_ARC_JUMP_SLOT)
    *reloc_addr += l_addr;
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

#endif /* RESOLVE_MAP */
