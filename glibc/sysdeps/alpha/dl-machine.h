/* Machine-dependent ELF dynamic relocation inline functions.  Alpha version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@tamu.edu>.

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

/* This was written in the absence of an ABI -- don't expect
   it to remain unchanged.  */

#ifndef dl_machine_h
#define dl_machine_h 1

#define ELF_MACHINE_NAME "alpha"

#include <string.h>


/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK	0x120000000UL

/* Translate a processor specific dynamic tag to the index in l_info array.  */
#define DT_ALPHA(x) (DT_ALPHA_##x - DT_LOPROC + DT_NUM)

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf64_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_ALPHA;
}

/* Return the link-time address of _DYNAMIC.  The multiple-got-capable
   linker no longer allocates the first .got entry for this.  But not to
   worry, no special tricks are needed.  */
static inline Elf64_Addr
elf_machine_dynamic (void)
{
#ifndef NO_AXP_MULTI_GOT_LD
  return (Elf64_Addr) &_DYNAMIC;
#else
  register Elf64_Addr *gp __asm__ ("$29");
  return gp[-4096];
#endif
}

/* Return the run-time load address of the shared object.  */

static inline Elf64_Addr
elf_machine_load_address (void)
{
  /* This relies on the compiler using gp-relative addresses for static symbols.  */
  static void *dot = &dot;
  return (void *)&dot - dot;
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int
elf_machine_runtime_setup (struct link_map *map, int lazy, int profile)
{
  extern char _dl_runtime_resolve_new[] attribute_hidden;
  extern char _dl_runtime_profile_new[] attribute_hidden;
  extern char _dl_runtime_resolve_old[] attribute_hidden;
  extern char _dl_runtime_profile_old[] attribute_hidden;

  struct pltgot {
    char *resolve;
    struct link_map *link;
  };

  struct pltgot *pg;
  long secureplt;
  char *resolve;

  if (map->l_info[DT_JMPREL] == 0 || !lazy)
    return lazy;

  /* Check to see if we're using the read-only plt form.  */
  secureplt = map->l_info[DT_ALPHA(PLTRO)] != 0;

  /* If the binary uses the read-only secure plt format, PG points to
     the .got.plt section, which is the right place for ld.so to place
     its hooks.  Otherwise, PG is currently pointing at the start of
     the plt; the hooks go at offset 16.  */
  pg = (struct pltgot *) D_PTR (map, l_info[DT_PLTGOT]);
  pg += !secureplt;

  /* This function will be called to perform the relocation.  They're
     not declared as functions to convince the compiler to use gp
     relative relocations for them.  */
  if (secureplt)
    resolve = _dl_runtime_resolve_new;
  else
    resolve = _dl_runtime_resolve_old;

  if (__builtin_expect (profile, 0))
    {
      if (secureplt)
	resolve = _dl_runtime_profile_new;
      else
	resolve = _dl_runtime_profile_old;

      if (GLRO(dl_profile) && _dl_name_match_p (GLRO(dl_profile), map))
	{
	  /* This is the object we are looking for.  Say that we really
	     want profiling and the timers are started.  */
	  GL(dl_profile_map) = map;
	}
    }

  pg->resolve = resolve;
  pg->link = map;

  return lazy;
}

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_START asm ("\
	.section .text						\n\
	.set at							\n\
	.globl _start						\n\
	.ent _start						\n\
_start:								\n\
	.frame $31,0,$31,0					\n\
	br	$gp, 0f						\n\
0:	ldgp	$gp, 0($gp)					\n\
	.prologue 0						\n\
	/* Pass pointer to argument block to _dl_start.  */	\n\
	mov	$sp, $16					\n\
	bsr	$26, _dl_start		!samegp			\n\
	.end _start						\n\
	/* FALLTHRU */						\n\
	.globl _dl_start_user					\n\
	.ent _dl_start_user					\n\
_dl_start_user:							\n\
	.frame $31,0,$31,0					\n\
	.prologue 0						\n\
	/* Save the user entry point address in s0.  */		\n\
	mov	$0, $9						\n\
	/* See if we were run as a command with the executable	\n\
	   file name as an extra leading argument.  */		\n\
	ldah	$1, _dl_skip_args($gp)	!gprelhigh		\n\
	ldl	$1, _dl_skip_args($1)	!gprellow		\n\
	bne	$1, $fixup_stack				\n\
$fixup_stack_ret:						\n\
	/* The special initializer gets called with the stack	\n\
	   just as the application's entry point will see it;	\n\
	   it can switch stacks if it moves these contents	\n\
	   over.  */						\n\
" RTLD_START_SPECIAL_INIT "					\n\
	/* Call _dl_init(_dl_loaded, argc, argv, envp) to run	\n\
	   initializers.  */					\n\
	ldah	$16, _rtld_local($gp)	!gprelhigh		\n\
	ldq	$16, _rtld_local($16)	!gprellow		\n\
	ldq	$17, 0($sp)					\n\
	lda	$18, 8($sp)					\n\
	s8addq	$17, 8, $19					\n\
	addq	$19, $18, $19					\n\
	bsr	$26, _dl_init		!samegp			\n\
	/* Pass our finalizer function to the user in $0. */	\n\
	ldah	$0, _dl_fini($gp)	!gprelhigh		\n\
	lda	$0, _dl_fini($0)	!gprellow		\n\
	/* Jump to the user's entry point.  */			\n\
	mov	$9, $27						\n\
	jmp	($9)						\n\
$fixup_stack:							\n\
	/* Adjust the stack pointer to skip _dl_skip_args words.\n\
	   This involves copying everything down, since the	\n\
	   stack pointer must always be 16-byte aligned.  */	\n\
	ldah	$7, __GI__dl_argv($gp) !gprelhigh		\n\
	ldq	$2, 0($sp)					\n\
	ldq	$5, __GI__dl_argv($7) !gprellow			\n\
	subq	$31, $1, $6					\n\
	subq	$2, $1, $2					\n\
	s8addq	$6, $5, $5					\n\
	mov	$sp, $4						\n\
	s8addq	$1, $sp, $3					\n\
	stq	$2, 0($sp)					\n\
	stq	$5, __GI__dl_argv($7) !gprellow			\n\
	/* Copy down argv.  */					\n\
0:	ldq	$5, 8($3)					\n\
	addq	$4, 8, $4					\n\
	addq	$3, 8, $3					\n\
	stq	$5, 0($4)					\n\
	bne	$5, 0b						\n\
	/* Copy down envp.  */					\n\
1:	ldq	$5, 8($3)					\n\
	addq	$4, 8, $4					\n\
	addq	$3, 8, $3					\n\
	stq	$5, 0($4)					\n\
	bne	$5, 1b						\n\
	/* Copy down auxiliary table.  */			\n\
2:	ldq	$5, 8($3)					\n\
	ldq	$6, 16($3)					\n\
	addq	$4, 16, $4					\n\
	addq	$3, 16, $3					\n\
	stq	$5, -8($4)					\n\
	stq	$6, 0($4)					\n\
	bne	$5, 2b						\n\
	br	$fixup_stack_ret				\n\
	.end _dl_start_user					\n\
	.set noat						\n\
.previous");

#ifndef RTLD_START_SPECIAL_INIT
#define RTLD_START_SPECIAL_INIT /* nothing */
#endif

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry
   or TLS variables, so undefined references should not be allowed
   to define the value.

   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve
   to one of the main executable's symbols, as for a COPY reloc.
   This is unused on Alpha.  */

# define elf_machine_type_class(type)	\
  (((type) == R_ALPHA_JMP_SLOT		\
    || (type) == R_ALPHA_DTPMOD64	\
    || (type) == R_ALPHA_DTPREL64	\
    || (type) == R_ALPHA_TPREL64) * ELF_RTYPE_CLASS_PLT)

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	 R_ALPHA_JMP_SLOT

/* The alpha never uses Elf64_Rel relocations.  */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

/* We define an initialization functions.  This is called very early in
 *    _dl_sysdep_start.  */
#define DL_PLATFORM_INIT dl_platform_init ()

static inline void __attribute__ ((unused))
dl_platform_init (void)
{
	if (GLRO(dl_platform) != NULL && *GLRO(dl_platform) == '\0')
	/* Avoid an empty string which would disturb us.  */
		GLRO(dl_platform) = NULL;
}

/* Fix up the instructions of a PLT entry to invoke the function
   rather than the dynamic linker.  */
static inline Elf64_Addr
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf64_Rela *reloc,
		       Elf64_Addr *got_addr, Elf64_Addr value)
{
  const Elf64_Rela *rela_plt;
  Elf64_Word *plte;
  long int edisp;

  /* Store the value we are going to load.  */
  *got_addr = value;

  /* If this binary uses the read-only secure plt format, we're done.  */
  if (map->l_info[DT_ALPHA(PLTRO)])
    return value;

  /* Otherwise we have to modify the plt entry in place to do the branch.  */

  /* Recover the PLT entry address by calculating reloc's index into the
     .rela.plt, and finding that entry in the .plt.  */
  rela_plt = (const Elf64_Rela *) D_PTR (map, l_info[DT_JMPREL]);
  plte = (Elf64_Word *) (D_PTR (map, l_info[DT_PLTGOT]) + 32);
  plte += 3 * (reloc - rela_plt);

  /* Find the displacement from the plt entry to the function.  */
  edisp = (long int) (value - (Elf64_Addr)&plte[3]) / 4;

  if (edisp >= -0x100000 && edisp < 0x100000)
    {
      /* If we are in range, use br to perfect branch prediction and
	 elide the dependency on the address load.  This case happens,
	 e.g., when a shared library call is resolved to the same library.  */

      int hi, lo;
      hi = value - (Elf64_Addr)&plte[0];
      lo = (short int) hi;
      hi = (hi - lo) >> 16;

      /* Emit "lda $27,lo($27)" */
      plte[1] = 0x237b0000 | (lo & 0xffff);

      /* Emit "br $31,function" */
      plte[2] = 0xc3e00000 | (edisp & 0x1fffff);

      /* Think about thread-safety -- the previous instructions must be
	 committed to memory before the first is overwritten.  */
      __asm__ __volatile__("wmb" : : : "memory");

      /* Emit "ldah $27,hi($27)" */
      plte[0] = 0x277b0000 | (hi & 0xffff);
    }
  else
    {
      /* Don't bother with the hint since we already know the hint is
	 wrong.  Eliding it prevents the wrong page from getting pulled
	 into the cache.  */

      int hi, lo;
      hi = (Elf64_Addr)got_addr - (Elf64_Addr)&plte[0];
      lo = (short)hi;
      hi = (hi - lo) >> 16;

      /* Emit "ldq $27,lo($27)" */
      plte[1] = 0xa77b0000 | (lo & 0xffff);

      /* Emit "jmp $31,($27)" */
      plte[2] = 0x6bfb0000;

      /* Think about thread-safety -- the previous instructions must be
	 committed to memory before the first is overwritten.  */
      __asm__ __volatile__("wmb" : : : "memory");

      /* Emit "ldah $27,hi($27)" */
      plte[0] = 0x277b0000 | (hi & 0xffff);
    }

  /* At this point, if we've been doing runtime resolution, Icache is dirty.
     This will be taken care of in _dl_runtime_resolve.  If instead we are
     doing this as part of non-lazy startup relocation, that bit of code
     hasn't made it into Icache yet, so there's nothing to clean up.  */

  return value;
}

/* Return the final value of a plt relocation.  */
static inline Elf64_Addr
elf_machine_plt_value (struct link_map *map, const Elf64_Rela *reloc,
		       Elf64_Addr value)
{
  return value + reloc->r_addend;
}

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER	alpha_gnu_pltenter
#define ARCH_LA_PLTEXIT		alpha_gnu_pltexit

#endif /* !dl_machine_h */

#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */
auto inline void
__attribute__ ((always_inline))
elf_machine_rela (struct link_map *map,
		  const Elf64_Rela *reloc,
		  const Elf64_Sym *sym,
		  const struct r_found_version *version,
		  void *const reloc_addr_arg,
		  int skip_ifunc)
{
  Elf64_Addr *const reloc_addr = reloc_addr_arg;
  unsigned long int const r_type = ELF64_R_TYPE (reloc->r_info);

#if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC && !defined SHARED
  /* This is defined in rtld.c, but nowhere in the static libc.a; make the
     reference weak so static programs can still link.  This declaration
     cannot be done when compiling rtld.c (i.e.  #ifdef RTLD_BOOTSTRAP)
     because rtld.c contains the common defn for _dl_rtld_map, which is
     incompatible with a weak decl in the same file.  */
  weak_extern (_dl_rtld_map);
#endif

  /* We cannot use a switch here because we cannot locate the switch
     jump table until we've self-relocated.  */

#if !defined RTLD_BOOTSTRAP || !defined HAVE_Z_COMBRELOC
  if (__builtin_expect (r_type == R_ALPHA_RELATIVE, 0))
    {
# if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC
      /* Already done in dynamic linker.  */
      if (map != &GL(dl_rtld_map))
# endif
	{
	  /* XXX Make some timings.  Maybe it's preferable to test for
	     unaligned access and only do it the complex way if necessary.  */
	  Elf64_Addr reloc_addr_val;

	  /* Load value without causing unaligned trap. */
	  memcpy (&reloc_addr_val, reloc_addr_arg, 8);
	  reloc_addr_val += map->l_addr;

	  /* Store value without causing unaligned trap. */
	  memcpy (reloc_addr_arg, &reloc_addr_val, 8);
	}
    }
  else
#endif
    if (__builtin_expect (r_type == R_ALPHA_NONE, 0))
      return;
  else
    {
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf64_Addr sym_value;
      Elf64_Addr sym_raw_value;

      sym_raw_value = sym_value = reloc->r_addend;
      if (sym_map)
	{
	  sym_raw_value += sym->st_value;
	  sym_value += SYMBOL_ADDRESS (sym_map, sym, true);
	}

      if (r_type == R_ALPHA_GLOB_DAT)
	*reloc_addr = sym_value;
#ifdef RESOLVE_CONFLICT_FIND_MAP
      /* In .gnu.conflict section, R_ALPHA_JMP_SLOT relocations have
	 R_ALPHA_JMP_SLOT in lower 8 bits and the remaining 24 bits
	 are .rela.plt index.  */
      else if ((r_type & 0xff) == R_ALPHA_JMP_SLOT)
	{
	  /* elf_machine_fixup_plt needs the map reloc_addr points into,
	     while in _dl_resolve_conflicts map is _dl_loaded.  */
	  RESOLVE_CONFLICT_FIND_MAP (map, reloc_addr);
	  reloc = ((const Elf64_Rela *) D_PTR (map, l_info[DT_JMPREL]))
		  + (r_type >> 8);
	  elf_machine_fixup_plt (map, 0, 0, 0, reloc, reloc_addr, sym_value);
	}
#else
      else if (r_type == R_ALPHA_JMP_SLOT)
	elf_machine_fixup_plt (map, 0, 0, 0, reloc, reloc_addr, sym_value);
#endif
#ifndef RTLD_BOOTSTRAP
      else if (r_type == R_ALPHA_REFQUAD)
	{
	  /* Store value without causing unaligned trap.  */
	  memcpy (reloc_addr_arg, &sym_value, 8);
	}
#endif
      else if (r_type == R_ALPHA_DTPMOD64)
	{
# ifdef RTLD_BOOTSTRAP
	  /* During startup the dynamic linker is always index 1.  */
	  *reloc_addr = 1;
# else
	  /* Get the information from the link map returned by the
	     resolv function.  */
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
# endif
	}
      else if (r_type == R_ALPHA_DTPREL64)
	{
# ifndef RTLD_BOOTSTRAP
	  /* During relocation all TLS symbols are defined and used.
	     Therefore the offset is already correct.  */
	  *reloc_addr = sym_raw_value;
# endif
	}
      else if (r_type == R_ALPHA_TPREL64)
	{
# ifdef RTLD_BOOTSTRAP
	  *reloc_addr = sym_raw_value + map->l_tls_offset;
# else
	  if (sym_map)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = sym_raw_value + sym_map->l_tls_offset;
	    }
# endif
	}
      else
	_dl_reloc_bad_type (map, r_type, 0);
    }
}

/* Let do-rel.h know that on Alpha if l_addr is 0, all RELATIVE relocs
   can be skipped.  */
#define ELF_MACHINE_REL_RELATIVE 1

auto inline void
__attribute__ ((always_inline))
elf_machine_rela_relative (Elf64_Addr l_addr, const Elf64_Rela *reloc,
			   void *const reloc_addr_arg)
{
  /* XXX Make some timings.  Maybe it's preferable to test for
     unaligned access and only do it the complex way if necessary.  */
  Elf64_Addr reloc_addr_val;

  /* Load value without causing unaligned trap. */
  memcpy (&reloc_addr_val, reloc_addr_arg, 8);
  reloc_addr_val += l_addr;

  /* Store value without causing unaligned trap. */
  memcpy (reloc_addr_arg, &reloc_addr_val, 8);
}

auto inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf64_Addr l_addr, const Elf64_Rela *reloc,
		      int skip_ifunc)
{
  Elf64_Addr * const reloc_addr = (void *)(l_addr + reloc->r_offset);
  unsigned long int const r_type = ELF64_R_TYPE (reloc->r_info);

  if (r_type == R_ALPHA_JMP_SLOT)
    {
      /* Perform a RELATIVE reloc on the .got entry that transfers
	 to the .plt.  */
      *reloc_addr += l_addr;
    }
  else if (r_type == R_ALPHA_NONE)
    return;
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

#endif /* RESOLVE_MAP */
