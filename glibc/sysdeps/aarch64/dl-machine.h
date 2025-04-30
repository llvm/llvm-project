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

#define ELF_MACHINE_NAME "aarch64"

#include <sysdep.h>
#include <tls.h>
#include <dl-tlsdesc.h>
#include <dl-irel.h>
#include <cpu-features.c>

/* Translate a processor specific dynamic tag to the index in l_info array.  */
#define DT_AARCH64(x) (DT_AARCH64_##x - DT_LOPROC + DT_NUM)

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int __attribute__ ((unused))
elf_machine_matches_host (const ElfW(Ehdr) *ehdr)
{
  return ehdr->e_machine == EM_AARCH64;
}

/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT. */
static inline ElfW(Addr) __attribute__ ((unused))
elf_machine_dynamic (void)
{
  extern const ElfW(Addr) _GLOBAL_OFFSET_TABLE_[] attribute_hidden;
  return _GLOBAL_OFFSET_TABLE_[0];
}

/* Return the run-time load address of the shared object.  */

static inline ElfW(Addr) __attribute__ ((unused))
elf_machine_load_address (void)
{
  /* To figure out the load address we use the definition that for any symbol:
     dynamic_addr(symbol) = static_addr(symbol) + load_addr

    _DYNAMIC sysmbol is used here as its link-time address stored in
    the special unrelocated first GOT entry.  */

    extern ElfW(Dyn) _DYNAMIC[] attribute_hidden;
    return (ElfW(Addr)) &_DYNAMIC - elf_machine_dynamic ();
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((unused))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  if (l->l_info[DT_JMPREL] && lazy)
    {
      ElfW(Addr) *got;
      extern void _dl_runtime_resolve (ElfW(Word));
      extern void _dl_runtime_profile (ElfW(Word));

      got = (ElfW(Addr) *) D_PTR (l, l_info[DT_PLTGOT]);
      if (got[1])
	{
	  l->l_mach.plt = got[1] + l->l_addr;
	}
      got[1] = (ElfW(Addr)) l;

      /* The got[2] entry contains the address of a function which gets
	 called to get the address of a so far unresolved function and
	 jump to it.  The profiling extension of the dynamic linker allows
	 to intercept the calls to collect information.  In this case we
	 don't store the address in the GOT so that all future calls also
	 end in this function.  */
      if ( profile)
	{
	   got[2] = (ElfW(Addr)) &_dl_runtime_profile;

	  if (GLRO(dl_profile) != NULL
	      && _dl_name_match_p (GLRO(dl_profile), l))
	    /* Say that we really want profiling and the timers are
	       started.  */
	    GL(dl_profile_map) = l;
	}
      else
	{
	  /* This function will get called to fix up the GOT entry
	     indicated by the offset on the stack, and then jump to
	     the resolved address.  */
	  got[2] = (ElfW(Addr)) &_dl_runtime_resolve;
	}
    }

  return lazy;
}

/* Initial entry point for the dynamic linker. The C function
   _dl_start is the real entry point, its return value is the user
   program's entry point */
#ifdef __LP64__
# define RTLD_START RTLD_START_1 ("x", "3", "sp")
#else
# define RTLD_START RTLD_START_1 ("w", "2", "wsp")
#endif


#define RTLD_START_1(PTR, PTR_SIZE_LOG, PTR_SP) asm ("\
.text									\n\
.globl _start								\n\
.type _start, %function							\n\
.globl _dl_start_user							\n\
.type _dl_start_user, %function						\n\
_start:									\n\
	// bti c							\n\
	hint	34							\n\
	mov	" PTR "0, " PTR_SP "					\n\
	bl	_dl_start						\n\
	// returns user entry point in x0				\n\
	mov	x21, x0							\n\
_dl_start_user:								\n\
	// get the original arg count					\n\
	ldr	" PTR "1, [sp]						\n\
	// get the argv address						\n\
	add	" PTR "2, " PTR_SP ", #(1<<"  PTR_SIZE_LOG ")		\n\
	// get _dl_skip_args to see if we were				\n\
	// invoked as an executable					\n\
	adrp	x4, _dl_skip_args					\n\
        ldr	w4, [x4, #:lo12:_dl_skip_args]				\n\
	// do we need to adjust argc/argv				\n\
        cmp	w4, 0							\n\
	beq	.L_done_stack_adjust					\n\
	// subtract _dl_skip_args from original arg count		\n\
	sub	" PTR "1, " PTR "1, " PTR "4				\n\
	// store adjusted argc back to stack				\n\
	str	" PTR "1, [sp]						\n\
	// find the first unskipped argument				\n\
	mov	" PTR "3, " PTR "2					\n\
	add	" PTR "4, " PTR "2, " PTR "4, lsl #" PTR_SIZE_LOG "	\n\
	// shuffle argv down						\n\
1:	ldr	" PTR "5, [x4], #(1<<"  PTR_SIZE_LOG ")			\n\
	str	" PTR "5, [x3], #(1<<"  PTR_SIZE_LOG ")			\n\
	cmp	" PTR "5, #0						\n\
	bne	1b							\n\
	// shuffle envp down						\n\
1:	ldr	" PTR "5, [x4], #(1<<"  PTR_SIZE_LOG ")			\n\
	str	" PTR "5, [x3], #(1<<"  PTR_SIZE_LOG ")			\n\
	cmp	" PTR "5, #0						\n\
	bne	1b							\n\
	// shuffle auxv down						\n\
1:	ldp	" PTR "0, " PTR "5, [x4, #(2<<"  PTR_SIZE_LOG ")]!	\n\
	stp	" PTR "0, " PTR "5, [x3], #(2<<"  PTR_SIZE_LOG ")	\n\
	cmp	" PTR "0, #0						\n\
	bne	1b							\n\
	// Update _dl_argv						\n\
	adrp	x3, __GI__dl_argv					\n\
	str	" PTR "2, [x3, #:lo12:__GI__dl_argv]			\n\
.L_done_stack_adjust:							\n\
	// compute envp							\n\
	add	" PTR "3, " PTR "2, " PTR "1, lsl #" PTR_SIZE_LOG "	\n\
	add	" PTR "3, " PTR "3, #(1<<"  PTR_SIZE_LOG ")		\n\
	adrp	x16, _rtld_local					\n\
        add	" PTR "16, " PTR "16, #:lo12:_rtld_local		\n\
        ldr	" PTR "0, [x16]						\n\
	bl	_dl_init						\n\
	// load the finalizer function					\n\
	adrp	x0, _dl_fini						\n\
	add	" PTR "0, " PTR "0, #:lo12:_dl_fini			\n\
	// jump to the user_s entry point				\n\
	mov     x16, x21						\n\
	br      x16							\n\
");

#define elf_machine_type_class(type)					\
  ((((type) == AARCH64_R(JUMP_SLOT)					\
     || (type) == AARCH64_R(TLS_DTPMOD)					\
     || (type) == AARCH64_R(TLS_DTPREL)					\
     || (type) == AARCH64_R(TLS_TPREL)					\
     || (type) == AARCH64_R(TLSDESC)) * ELF_RTYPE_CLASS_PLT)		\
   | (((type) == AARCH64_R(COPY)) * ELF_RTYPE_CLASS_COPY)		\
   | (((type) == AARCH64_R(GLOB_DAT)) * ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA))

#define ELF_MACHINE_JMP_SLOT	AARCH64_R(JUMP_SLOT)

/* AArch64 uses RELA not REL */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

#define DL_PLATFORM_INIT dl_platform_init ()

static inline void __attribute__ ((unused))
dl_platform_init (void)
{
  if (GLRO(dl_platform) != NULL && *GLRO(dl_platform) == '\0')
    /* Avoid an empty string which would disturb us.  */
    GLRO(dl_platform) = NULL;

#ifdef SHARED
  /* init_cpu_features has been called early from __libc_start_main in
     static executable.  */
  init_cpu_features (&GLRO(dl_aarch64_cpu_features));
#endif
}


static inline ElfW(Addr)
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const ElfW(Rela) *reloc,
		       ElfW(Addr) *reloc_addr,
		       ElfW(Addr) value)
{
  return *reloc_addr = value;
}

/* Return the final value of a plt relocation.  */
static inline ElfW(Addr)
elf_machine_plt_value (struct link_map *map,
		       const ElfW(Rela) *reloc,
		       ElfW(Addr) value)
{
  return value;
}

#endif

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER aarch64_gnu_pltenter
#define ARCH_LA_PLTEXIT  aarch64_gnu_pltexit

#ifdef RESOLVE_MAP

auto inline void
__attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const ElfW(Rela) *reloc,
		  const ElfW(Sym) *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  ElfW(Addr) *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELFW (R_TYPE) (reloc->r_info);

  if (__builtin_expect (r_type == AARCH64_R(RELATIVE), 0))
      *reloc_addr = map->l_addr + reloc->r_addend;
  else if (__builtin_expect (r_type == R_AARCH64_NONE, 0))
      return;
  else
    {
      const ElfW(Sym) *const refsym = sym;
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      ElfW(Addr) value = SYMBOL_ADDRESS (sym_map, sym, true);

      if (sym != NULL
	  && __glibc_unlikely (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC)
	  && __glibc_likely (sym->st_shndx != SHN_UNDEF)
	  && __glibc_likely (!skip_ifunc))
	value = elf_ifunc_invoke (value);

      switch (r_type)
	{
	case AARCH64_R(COPY):
	  if (sym == NULL)
	      break;

	  if (sym->st_size > refsym->st_size
	      || (GLRO(dl_verbose) && sym->st_size < refsym->st_size))
	    {
	      const char *strtab;

	      strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
	      _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				RTLD_PROGNAME, strtab + refsym->st_name);
	    }
	  memcpy (reloc_addr_arg, (void *) value,
		  sym->st_size < refsym->st_size
		  ? sym->st_size : refsym->st_size);
	  break;

	case AARCH64_R(RELATIVE):
	case AARCH64_R(GLOB_DAT):
	case AARCH64_R(JUMP_SLOT):
	case AARCH64_R(ABS32):
#ifdef __LP64__
	case AARCH64_R(ABS64):
#endif
	  *reloc_addr = value + reloc->r_addend;
	  break;

	case AARCH64_R(TLSDESC):
	  {
	    struct tlsdesc volatile *td =
	      (struct tlsdesc volatile *)reloc_addr;
#ifndef RTLD_BOOTSTRAP
	    if (! sym)
	      {
		td->arg = (void*)reloc->r_addend;
		td->entry = _dl_tlsdesc_undefweak;
	      }
	    else
#endif
	      {
#ifndef RTLD_BOOTSTRAP
# ifndef SHARED
		CHECK_STATIC_TLS (map, sym_map);
# else
		if (!TRY_STATIC_TLS (map, sym_map))
		  {
		    td->arg = _dl_make_tlsdesc_dynamic
		      (sym_map, sym->st_value + reloc->r_addend);
		    td->entry = _dl_tlsdesc_dynamic;
		  }
		else
# endif
#endif
		  {
		    td->arg = (void*)(sym->st_value + sym_map->l_tls_offset
				      + reloc->r_addend);
		    td->entry = _dl_tlsdesc_return;
		  }
	      }
	    break;
	  }

	case AARCH64_R(TLS_DTPMOD):
#ifdef RTLD_BOOTSTRAP
	  *reloc_addr = 1;
#else
	  if (sym_map != NULL)
	    {
	      *reloc_addr = sym_map->l_tls_modid;
	    }
#endif
	  break;

	case AARCH64_R(TLS_DTPREL):
	  if (sym)
	    *reloc_addr = sym->st_value + reloc->r_addend;
	  break;

	case AARCH64_R(TLS_TPREL):
	  if (sym)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr =
		sym->st_value + reloc->r_addend + sym_map->l_tls_offset;
	    }
	  break;

	case AARCH64_R(IRELATIVE):
	  value = map->l_addr + reloc->r_addend;
	  if (__glibc_likely (!skip_ifunc))
	    value = elf_ifunc_invoke (value);
	  *reloc_addr = value;
	  break;

	default:
	  _dl_reloc_bad_type (map, r_type, 0);
	  break;
	}
    }
}

inline void
__attribute__ ((always_inline))
elf_machine_rela_relative (ElfW(Addr) l_addr,
			   const ElfW(Rela) *reloc,
			   void *const reloc_addr_arg)
{
  ElfW(Addr) *const reloc_addr = reloc_addr_arg;
  *reloc_addr = l_addr + reloc->r_addend;
}

inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      ElfW(Addr) l_addr,
		      const ElfW(Rela) *reloc,
		      int skip_ifunc)
{
  ElfW(Addr) *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELFW (R_TYPE) (reloc->r_info);
  /* Check for unexpected PLT reloc type.  */
  if (__builtin_expect (r_type == AARCH64_R(JUMP_SLOT), 1))
    {
      if (__glibc_unlikely (map->l_info[DT_AARCH64 (VARIANT_PCS)] != NULL))
	{
	  /* Check the symbol table for variant PCS symbols.  */
	  const Elf_Symndx symndx = ELFW (R_SYM) (reloc->r_info);
	  const ElfW (Sym) *symtab =
	    (const void *)D_PTR (map, l_info[DT_SYMTAB]);
	  const ElfW (Sym) *sym = &symtab[symndx];
	  if (__glibc_unlikely (sym->st_other & STO_AARCH64_VARIANT_PCS))
	    {
	      /* Avoid lazy resolution of variant PCS symbols.  */
	      const struct r_found_version *version = NULL;
	      if (map->l_info[VERSYMIDX (DT_VERSYM)] != NULL)
		{
		  const ElfW (Half) *vernum =
		    (const void *)D_PTR (map, l_info[VERSYMIDX (DT_VERSYM)]);
		  version = &map->l_versions[vernum[symndx] & 0x7fff];
		}
	      elf_machine_rela (map, reloc, sym, version, reloc_addr,
				skip_ifunc);
	      return;
	    }
	}

      if (map->l_mach.plt == 0)
	*reloc_addr += l_addr;
      else
	*reloc_addr = map->l_mach.plt;
    }
  else if (__builtin_expect (r_type == AARCH64_R(TLSDESC), 1))
    {
      const Elf_Symndx symndx = ELFW (R_SYM) (reloc->r_info);
      const ElfW (Sym) *symtab = (const void *)D_PTR (map, l_info[DT_SYMTAB]);
      const ElfW (Sym) *sym = &symtab[symndx];
      const struct r_found_version *version = NULL;

      if (map->l_info[VERSYMIDX (DT_VERSYM)] != NULL)
	{
	  const ElfW (Half) *vernum =
	    (const void *)D_PTR (map, l_info[VERSYMIDX (DT_VERSYM)]);
	  version = &map->l_versions[vernum[symndx] & 0x7fff];
	}

      /* Always initialize TLS descriptors completely, because lazy
	 initialization requires synchronization at every TLS access.  */
      elf_machine_rela (map, reloc, sym, version, reloc_addr, skip_ifunc);
    }
  else if (__glibc_unlikely (r_type == AARCH64_R(IRELATIVE)))
    {
      ElfW(Addr) value = map->l_addr + reloc->r_addend;
      if (__glibc_likely (!skip_ifunc))
	value = elf_ifunc_invoke (value);
      *reloc_addr = value;
    }
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

#endif
