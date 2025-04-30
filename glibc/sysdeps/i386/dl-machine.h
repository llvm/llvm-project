/* Machine-dependent ELF dynamic relocation inline functions.  i386 version.
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

#define ELF_MACHINE_NAME "i386"

#include <sys/param.h>
#include <sysdep.h>
#include <tls.h>
#include <dl-tlsdesc.h>

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int __attribute__ ((unused))
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_386;
}


/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT, a special entry that is never relocated.  */
static inline Elf32_Addr __attribute__ ((unused, const))
elf_machine_dynamic (void)
{
  /* This produces a GOTOFF reloc that resolves to zero at link time, so in
     fact just loads from the GOT register directly.  By doing it without
     an asm we can let the compiler choose any register.  */
  extern const Elf32_Addr _GLOBAL_OFFSET_TABLE_[] attribute_hidden;
  return _GLOBAL_OFFSET_TABLE_[0];
}

/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr __attribute__ ((unused))
elf_machine_load_address (void)
{
  /* Compute the difference between the runtime address of _DYNAMIC as seen
     by a GOTOFF reference, and the link-time address found in the special
     unrelocated first GOT entry.  */
  extern Elf32_Dyn bygotoff[] asm ("_DYNAMIC") attribute_hidden;
  return (Elf32_Addr) &bygotoff - elf_machine_dynamic ();
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((unused, always_inline))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  Elf32_Addr *got;
  extern void _dl_runtime_resolve (Elf32_Word) attribute_hidden;
  extern void _dl_runtime_profile (Elf32_Word) attribute_hidden;
  extern void _dl_runtime_resolve_shstk (Elf32_Word) attribute_hidden;
  extern void _dl_runtime_profile_shstk (Elf32_Word) attribute_hidden;
  /* Check if SHSTK is enabled by kernel.  */
  bool shstk_enabled
    = (GL(dl_x86_feature_1) & GNU_PROPERTY_X86_FEATURE_1_SHSTK) != 0;

  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* The GOT entries for functions in the PLT have not yet been filled
	 in.  Their initial contents will arrange when called to push an
	 offset into the .rel.plt section, push _GLOBAL_OFFSET_TABLE_[1],
	 and then jump to _GLOBAL_OFFSET_TABLE[2].  */
      got = (Elf32_Addr *) D_PTR (l, l_info[DT_PLTGOT]);
      /* If a library is prelinked but we have to relocate anyway,
	 we have to be able to undo the prelinking of .got.plt.
	 The prelinker saved us here address of .plt + 0x16.  */
      if (got[1])
	{
	  l->l_mach.plt = got[1] + l->l_addr;
	  l->l_mach.gotplt = (Elf32_Addr) &got[3];
	}
      got[1] = (Elf32_Addr) l;	/* Identify this shared object.  */

      /* The got[2] entry contains the address of a function which gets
	 called to get the address of a so far unresolved function and
	 jump to it.  The profiling extension of the dynamic linker allows
	 to intercept the calls to collect information.  In this case we
	 don't store the address in the GOT so that all future calls also
	 end in this function.  */
      if (__glibc_unlikely (profile))
	{
	  got[2] = (shstk_enabled
		    ? (Elf32_Addr) &_dl_runtime_profile_shstk
		    : (Elf32_Addr) &_dl_runtime_profile);

	  if (GLRO(dl_profile) != NULL
	      && _dl_name_match_p (GLRO(dl_profile), l))
	    /* This is the object we are looking for.  Say that we really
	       want profiling and the timers are started.  */
	    GL(dl_profile_map) = l;
	}
      else
	/* This function will get called to fix up the GOT entry indicated by
	   the offset on the stack, and then jump to the resolved address.  */
	got[2] = (shstk_enabled
		  ? (Elf32_Addr) &_dl_runtime_resolve_shstk
		  : (Elf32_Addr) &_dl_runtime_resolve);
    }

  return lazy;
}

#ifdef IN_DL_RUNTIME

# ifndef PROF
/* We add a declaration of this function here so that in dl-runtime.c
   the ELF_MACHINE_RUNTIME_TRAMPOLINE macro really can pass the parameters
   in registers.

   We cannot use this scheme for profiling because the _mcount call
   destroys the passed register information.  */
#define ARCH_FIXUP_ATTRIBUTE __attribute__ ((regparm (3), stdcall, unused))

extern ElfW(Addr) _dl_fixup (struct link_map *l,
			     ElfW(Word) reloc_offset)
     ARCH_FIXUP_ATTRIBUTE;
extern ElfW(Addr) _dl_profile_fixup (struct link_map *l,
				     ElfW(Word) reloc_offset,
				     ElfW(Addr) retaddr, void *regs,
				     long int *framesizep)
     ARCH_FIXUP_ATTRIBUTE;
# endif

#endif

/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK	0xf0000000UL

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_START asm ("\n\
	.text\n\
	.align 16\n\
0:	movl (%esp), %ebx\n\
	ret\n\
	.align 16\n\
.globl _start\n\
.globl _dl_start_user\n\
_start:\n\
	movl %esp, %eax\n\
        subl $12, %esp\n\
        pushl %eax\n\
	call _dl_start\n\
        addl $16, %esp\n\
_dl_start_user:\n\
	# Save the user entry point address in %edi.\n\
	movl %eax, %edi\n\
	# Point %ebx at the GOT.\n\
	call 0b\n\
	addl $_GLOBAL_OFFSET_TABLE_, %ebx\n\
	# See if we were run as a command with the executable file\n\
	# name as an extra leading argument.\n\
	movl _dl_skip_args@GOTOFF(%ebx), %eax\n\
	# Pop the original argument count.\n\
	popl %edx\n\
	# Adjust the stack pointer to skip _dl_skip_args words.\n\
	leal (%esp,%eax,4), %esp\n\
	# Subtract _dl_skip_args from argc.\n\
	subl %eax, %edx\n\
	# Push argc back on the stack.\n\
	push %edx\n\
	# The special initializer gets called with the stack just\n\
	# as the application's entry point will see it; it can\n\
	# switch stacks if it moves these contents over.\n\
" RTLD_START_SPECIAL_INIT "\n\
	# Load the parameters again.\n\
	# (eax, edx, ecx, esi) = (_dl_loaded, argc, argv, envp)\n\
	movl _rtld_local@GOTOFF(%ebx), %eax\n\
	leal 8(%esp,%edx,4), %esi\n\
	leal 4(%esp), %ecx\n\
	movl %esp, %ebp\n\
	# Make sure _dl_init is run with 16 byte aligned stack.\n\
	andl $-16, %esp\n\
        subl $12, %esp\n\
	pushl %ebp\n\
        # Arguments for _dl_init.\n\
	pushl %esi\n\
	pushl %ecx\n\
	pushl %edx\n\
	pushl %eax\n\
	# Clear %ebp, so that even constructors have terminated backchain.\n\
	xorl %ebp, %ebp\n\
	# Call the function to run the initializers.\n\
	call _dl_init\n\
	# Pass our finalizer function to the user in %edx, as per ELF ABI.\n\
	leal _dl_fini@GOTOFF(%ebx), %edx\n\
	# Restore %esp _start expects.\n\
	movl 16(%esp), %esp\n\
	# Jump to the user's entry point.\n\
	jmp *%edi\n\
	.previous\n\
");

#ifndef RTLD_START_SPECIAL_INIT
# define RTLD_START_SPECIAL_INIT /* nothing */
#endif

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry or
   TLS variable, so undefined references should not be allowed to
   define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.
   ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA iff TYPE describes relocation may
   against protected data whose address be external due to copy relocation.
 */
# define elf_machine_type_class(type) \
  ((((type) == R_386_JMP_SLOT || (type) == R_386_TLS_DTPMOD32		      \
     || (type) == R_386_TLS_DTPOFF32 || (type) == R_386_TLS_TPOFF32	      \
     || (type) == R_386_TLS_TPOFF || (type) == R_386_TLS_DESC)		      \
    * ELF_RTYPE_CLASS_PLT)						      \
   | (((type) == R_386_COPY) * ELF_RTYPE_CLASS_COPY)			      \
   | (((type) == R_386_GLOB_DAT) * ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA))

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	R_386_JMP_SLOT

/* The i386 never uses Elf32_Rela relocations for the dynamic linker.
   Prelinked libraries may use Elf32_Rela though.  */
#define ELF_MACHINE_PLT_REL 1

/* We define an initialization functions.  This is called very early in
   _dl_sysdep_start.  */
#define DL_PLATFORM_INIT dl_platform_init ()

static inline void __attribute__ ((unused))
dl_platform_init (void)
{
#if IS_IN (rtld)
  /* _dl_x86_init_cpu_features is a wrapper for init_cpu_features which
     has been called early from __libc_start_main in static executable.  */
  _dl_x86_init_cpu_features ();
#else
  if (GLRO(dl_platform) != NULL && *GLRO(dl_platform) == '\0')
    /* Avoid an empty string which would disturb us.  */
    GLRO(dl_platform) = NULL;
#endif
}

static inline Elf32_Addr
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf32_Rel *reloc,
		       Elf32_Addr *reloc_addr, Elf32_Addr value)
{
  return *reloc_addr = value;
}

/* Return the final value of a plt relocation.  */
static inline Elf32_Addr
elf_machine_plt_value (struct link_map *map, const Elf32_Rel *reloc,
		       Elf32_Addr value)
{
  return value;
}


/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER i86_gnu_pltenter
#define ARCH_LA_PLTEXIT i86_gnu_pltexit

#endif /* !dl_machine_h */

/* The i386 never uses Elf32_Rela relocations for the dynamic linker.
   Prelinked libraries may use Elf32_Rela though.  */
#define ELF_MACHINE_NO_RELA defined RTLD_BOOTSTRAP
#define ELF_MACHINE_NO_REL 0

#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

auto inline void
__attribute ((always_inline))
elf_machine_rel (struct link_map *map, const Elf32_Rel *reloc,
		 const Elf32_Sym *sym, const struct r_found_version *version,
		 void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);

# if !defined RTLD_BOOTSTRAP || !defined HAVE_Z_COMBRELOC
  if (__glibc_unlikely (r_type == R_386_RELATIVE))
    {
#  if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC
      /* This is defined in rtld.c, but nowhere in the static libc.a;
	 make the reference weak so static programs can still link.
	 This declaration cannot be done when compiling rtld.c
	 (i.e. #ifdef RTLD_BOOTSTRAP) because rtld.c contains the
	 common defn for _dl_rtld_map, which is incompatible with a
	 weak decl in the same file.  */
#   ifndef SHARED
      weak_extern (_dl_rtld_map);
#   endif
      if (map != &GL(dl_rtld_map)) /* Already done in rtld itself.  */
#  endif
	*reloc_addr += map->l_addr;
    }
#  ifndef RTLD_BOOTSTRAP
  else if (__glibc_unlikely (r_type == R_386_NONE))
    return;
#  endif
  else
# endif	/* !RTLD_BOOTSTRAP and have no -z combreloc */
    {
# ifndef RTLD_BOOTSTRAP
      const Elf32_Sym *const refsym = sym;
# endif
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf32_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      if (sym != NULL
	  && __glibc_unlikely (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC)
	  && __glibc_likely (sym->st_shndx != SHN_UNDEF)
	  && __glibc_likely (!skip_ifunc))
	{
# ifndef RTLD_BOOTSTRAP
	  if (sym_map != map
	      && !sym_map->l_relocated)
	    {
	      const char *strtab
		= (const char *) D_PTR (map, l_info[DT_STRTAB]);
	      if (sym_map->l_type == lt_executable)
		_dl_fatal_printf ("\
%s: IFUNC symbol '%s' referenced in '%s' is defined in the executable \
and creates an unsatisfiable circular dependency.\n",
				  RTLD_PROGNAME, strtab + refsym->st_name,
				  map->l_name);
	      else
		_dl_error_printf ("\
%s: Relink `%s' with `%s' for IFUNC symbol `%s'\n",
				  RTLD_PROGNAME, map->l_name,
				  sym_map->l_name,
				  strtab + refsym->st_name);
	    }
# endif
	  value = ((Elf32_Addr (*) (void)) value) ();
	}

      switch (r_type)
	{
# ifndef RTLD_BOOTSTRAP
	case R_386_SIZE32:
	  /* Set to symbol size plus addend.  */
	  *reloc_addr += sym->st_size;
	  break;
# endif
	case R_386_GLOB_DAT:
	case R_386_JMP_SLOT:
	  *reloc_addr = value;
	  break;

	case R_386_TLS_DTPMOD32:
# ifdef RTLD_BOOTSTRAP
	  /* During startup the dynamic linker is always the module
	     with index 1.
	     XXX If this relocation is necessary move before RESOLVE
	     call.  */
	  *reloc_addr = 1;
# else
	  /* Get the information from the link map returned by the
	     resolv function.  */
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
# endif
	  break;
	case R_386_TLS_DTPOFF32:
# ifndef RTLD_BOOTSTRAP
	  /* During relocation all TLS symbols are defined and used.
	     Therefore the offset is already correct.  */
	  if (sym != NULL)
	    *reloc_addr = sym->st_value;
# endif
	  break;
	case R_386_TLS_DESC:
	  {
	    struct tlsdesc volatile *td =
	      (struct tlsdesc volatile *)reloc_addr;

# ifndef RTLD_BOOTSTRAP
	    if (! sym)
	      td->entry = _dl_tlsdesc_undefweak;
	    else
# endif
	      {
# ifndef RTLD_BOOTSTRAP
#  ifndef SHARED
		CHECK_STATIC_TLS (map, sym_map);
#  else
		if (!TRY_STATIC_TLS (map, sym_map))
		  {
		    td->arg = _dl_make_tlsdesc_dynamic
		      (sym_map, sym->st_value + (ElfW(Word))td->arg);
		    td->entry = _dl_tlsdesc_dynamic;
		  }
		else
#  endif
# endif
		  {
		    td->arg = (void*)(sym->st_value - sym_map->l_tls_offset
				      + (ElfW(Word))td->arg);
		    td->entry = _dl_tlsdesc_return;
		  }
	      }
	    break;
	  }
	case R_386_TLS_TPOFF32:
	  /* The offset is positive, backward from the thread pointer.  */
#  ifdef RTLD_BOOTSTRAP
	  *reloc_addr += map->l_tls_offset - sym->st_value;
#  else
	  /* We know the offset of object the symbol is contained in.
	     It is a positive value which will be subtracted from the
	     thread pointer.  To get the variable position in the TLS
	     block we subtract the offset from that of the TLS block.  */
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr += sym_map->l_tls_offset - sym->st_value;
	    }
# endif
	  break;
	case R_386_TLS_TPOFF:
	  /* The offset is negative, forward from the thread pointer.  */
# ifdef RTLD_BOOTSTRAP
	  *reloc_addr += sym->st_value - map->l_tls_offset;
# else
	  /* We know the offset of object the symbol is contained in.
	     It is a negative value which will be added to the
	     thread pointer.  */
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr += sym->st_value - sym_map->l_tls_offset;
	    }
# endif
	  break;

# ifndef RTLD_BOOTSTRAP
	case R_386_32:
	  *reloc_addr += value;
	  break;
	case R_386_PC32:
	  *reloc_addr += (value - (Elf32_Addr) reloc_addr);
	  break;
	case R_386_COPY:
	  if (sym == NULL)
	    /* This can happen in trace mode if an object could not be
	       found.  */
	    break;
	  if (__glibc_unlikely (sym->st_size > refsym->st_size)
	      || (__glibc_unlikely(sym->st_size < refsym->st_size)
		  && GLRO(dl_verbose)))
	    {
	      const char *strtab;

	      strtab = (const char *) D_PTR (map, l_info[DT_STRTAB]);
	      _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				RTLD_PROGNAME, strtab + refsym->st_name);
	    }
	  memcpy (reloc_addr_arg, (void *) value,
		  MIN (sym->st_size, refsym->st_size));
	  break;
	case R_386_IRELATIVE:
	  value = map->l_addr + *reloc_addr;
	  if (__glibc_likely (!skip_ifunc))
	    value = ((Elf32_Addr (*) (void)) value) ();
	  *reloc_addr = value;
	  break;
	default:
	  _dl_reloc_bad_type (map, r_type, 0);
	  break;
# endif	/* !RTLD_BOOTSTRAP */
	}
    }
}

# ifndef RTLD_BOOTSTRAP
auto inline void
__attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const Elf32_Rela *reloc,
		  const Elf32_Sym *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);

  if (ELF32_R_TYPE (reloc->r_info) == R_386_RELATIVE)
    *reloc_addr = map->l_addr + reloc->r_addend;
  else if (r_type != R_386_NONE)
    {
#  ifndef RESOLVE_CONFLICT_FIND_MAP
      const Elf32_Sym *const refsym = sym;
#  endif
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf32_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      if (sym != NULL
	  && __glibc_likely (sym->st_shndx != SHN_UNDEF)
	  && __glibc_unlikely (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC)
	  && __glibc_likely (!skip_ifunc))
	value = ((Elf32_Addr (*) (void)) value) ();

      switch (ELF32_R_TYPE (reloc->r_info))
	{
	case R_386_SIZE32:
	  /* Set to symbol size plus addend.  */
	  value = sym->st_size;
	  /* Fall through.  */
	case R_386_GLOB_DAT:
	case R_386_JMP_SLOT:
	case R_386_32:
	  *reloc_addr = value + reloc->r_addend;
	  break;
#  ifndef RESOLVE_CONFLICT_FIND_MAP
	  /* Not needed for dl-conflict.c.  */
	case R_386_PC32:
	  *reloc_addr = (value + reloc->r_addend - (Elf32_Addr) reloc_addr);
	  break;

	case R_386_TLS_DTPMOD32:
	  /* Get the information from the link map returned by the
	     resolv function.  */
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
	  break;
	case R_386_TLS_DTPOFF32:
	  /* During relocation all TLS symbols are defined and used.
	     Therefore the offset is already correct.  */
	  *reloc_addr = (sym == NULL ? 0 : sym->st_value) + reloc->r_addend;
	  break;
	case R_386_TLS_DESC:
	  {
	    struct tlsdesc volatile *td =
	      (struct tlsdesc volatile *)reloc_addr;

#   ifndef RTLD_BOOTSTRAP
	    if (!sym)
	      {
		td->arg = (void*)reloc->r_addend;
		td->entry = _dl_tlsdesc_undefweak;
	      }
	    else
#   endif
	      {
#   ifndef RTLD_BOOTSTRAP
#    ifndef SHARED
		CHECK_STATIC_TLS (map, sym_map);
#    else
		if (!TRY_STATIC_TLS (map, sym_map))
		  {
		    td->arg = _dl_make_tlsdesc_dynamic
		      (sym_map, sym->st_value + reloc->r_addend);
		    td->entry = _dl_tlsdesc_dynamic;
		  }
		else
#    endif
#   endif
		  {
		    td->arg = (void*)(sym->st_value - sym_map->l_tls_offset
				      + reloc->r_addend);
		    td->entry = _dl_tlsdesc_return;
		  }
	      }
	  }
	  break;
	case R_386_TLS_TPOFF32:
	  /* The offset is positive, backward from the thread pointer.  */
	  /* We know the offset of object the symbol is contained in.
	     It is a positive value which will be subtracted from the
	     thread pointer.  To get the variable position in the TLS
	     block we subtract the offset from that of the TLS block.  */
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = sym_map->l_tls_offset - sym->st_value
			    + reloc->r_addend;
	    }
	  break;
	case R_386_TLS_TPOFF:
	  /* The offset is negative, forward from the thread pointer.  */
	  /* We know the offset of object the symbol is contained in.
	     It is a negative value which will be added to the
	     thread pointer.  */
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = sym->st_value - sym_map->l_tls_offset
			    + reloc->r_addend;
	    }
	  break;
	case R_386_COPY:
	  if (sym == NULL)
	    /* This can happen in trace mode if an object could not be
	       found.  */
	    break;
	  if (__glibc_unlikely (sym->st_size > refsym->st_size)
	      || (__glibc_unlikely (sym->st_size < refsym->st_size)
		  && GLRO(dl_verbose)))
	    {
	      const char *strtab;

	      strtab = (const char *) D_PTR (map, l_info[DT_STRTAB]);
	      _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				RTLD_PROGNAME, strtab + refsym->st_name);
	    }
	  memcpy (reloc_addr_arg, (void *) value,
		  MIN (sym->st_size, refsym->st_size));
	  break;
#  endif /* !RESOLVE_CONFLICT_FIND_MAP */
	case R_386_IRELATIVE:
	  value = map->l_addr + reloc->r_addend;
	  if (__glibc_likely (!skip_ifunc))
	    value = ((Elf32_Addr (*) (void)) value) ();
	  *reloc_addr = value;
	  break;
	default:
	  /* We add these checks in the version to relocate ld.so only
	     if we are still debugging.  */
	  _dl_reloc_bad_type (map, r_type, 0);
	  break;
	}
    }
}
# endif	/* !RTLD_BOOTSTRAP */

auto inline void
__attribute ((always_inline))
elf_machine_rel_relative (Elf32_Addr l_addr, const Elf32_Rel *reloc,
			  void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  assert (ELF32_R_TYPE (reloc->r_info) == R_386_RELATIVE);
  *reloc_addr += l_addr;
}

# ifndef RTLD_BOOTSTRAP
auto inline void
__attribute__ ((always_inline))
elf_machine_rela_relative (Elf32_Addr l_addr, const Elf32_Rela *reloc,
			   void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  *reloc_addr = l_addr + reloc->r_addend;
}
# endif	/* !RTLD_BOOTSTRAP */

auto inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rel *reloc,
		      int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);
  /* Check for unexpected PLT reloc type.  */
  if (__glibc_likely (r_type == R_386_JMP_SLOT))
    {
      /* Prelink has been deprecated.  */
      if (__glibc_likely (map->l_mach.plt == 0))
	*reloc_addr += l_addr;
      else
	*reloc_addr = (map->l_mach.plt
		       + (((Elf32_Addr) reloc_addr) - map->l_mach.gotplt) * 4);
    }
  else if (__glibc_likely (r_type == R_386_TLS_DESC))
    {
      const Elf32_Rel *const r = reloc;
      /* The code below was borrowed from elf_dynamic_do_rel().  */
      const ElfW(Sym) *const symtab =
	(const void *) D_PTR (map, l_info[DT_SYMTAB]);

      /* Always initialize TLS descriptors completely at load time, in
	 case static TLS is allocated for it that requires locking.  */
# ifdef RTLD_BOOTSTRAP
      /* The dynamic linker always uses versioning.  */
      assert (map->l_info[VERSYMIDX (DT_VERSYM)] != NULL);
# else
      if (map->l_info[VERSYMIDX (DT_VERSYM)])
# endif
	{
	  const ElfW(Half) *const version =
	    (const void *) D_PTR (map, l_info[VERSYMIDX (DT_VERSYM)]);
	  ElfW(Half) ndx = version[ELFW(R_SYM) (r->r_info)] & 0x7fff;
	  elf_machine_rel (map, r, &symtab[ELFW(R_SYM) (r->r_info)],
			   &map->l_versions[ndx],
			   (void *) (l_addr + r->r_offset), skip_ifunc);
	}
# ifndef RTLD_BOOTSTRAP
      else
	elf_machine_rel (map, r, &symtab[ELFW(R_SYM) (r->r_info)], NULL,
			 (void *) (l_addr + r->r_offset), skip_ifunc);
# endif
    }
  else if (__glibc_unlikely (r_type == R_386_IRELATIVE))
    {
      Elf32_Addr value = map->l_addr + *reloc_addr;
      if (__glibc_likely (!skip_ifunc))
	value = ((Elf32_Addr (*) (void)) value) ();
      *reloc_addr = value;
    }
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

# ifndef RTLD_BOOTSTRAP

auto inline void
__attribute__ ((always_inline))
elf_machine_lazy_rela (struct link_map *map,
		       Elf32_Addr l_addr, const Elf32_Rela *reloc,
		       int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);
  if (__glibc_likely (r_type == R_386_JMP_SLOT))
    ;
  else if (__glibc_likely (r_type == R_386_TLS_DESC))
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

      /* Always initialize TLS descriptors completely at load time, in
	 case static TLS is allocated for it that requires locking.  */
      elf_machine_rela (map, reloc, sym, version, reloc_addr, skip_ifunc
#ifndef NESTING
          , NULL
#endif
      );
    }
  else if (__glibc_unlikely (r_type == R_386_IRELATIVE))
    {
      Elf32_Addr value = map->l_addr + reloc->r_addend;
      if (__glibc_likely (!skip_ifunc))
	value = ((Elf32_Addr (*) (void)) value) ();
      *reloc_addr = value;
    }
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

# endif	/* !RTLD_BOOTSTRAP */

#endif /* RESOLVE_MAP */
