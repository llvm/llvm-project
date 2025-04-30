/* Machine-dependent ELF dynamic relocation inline functions.
   64 bit S/390 Version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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

#define ELF_MACHINE_NAME "s390x"

#include <sys/param.h>
#include <string.h>
#include <link.h>
#include <sysdeps/s390/dl-procinfo.h>
#include <dl-irel.h>

#define ELF_MACHINE_IRELATIVE       R_390_IRELATIVE

/* This is an older, now obsolete value.  */
#define EM_S390_OLD	0xA390

/* Return nonzero iff E_MACHINE is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf64_Ehdr *ehdr)
{
  return (ehdr->e_machine == EM_S390 || ehdr->e_machine == EM_S390_OLD)
	 && ehdr->e_ident[EI_CLASS] == ELFCLASS64;
}

/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  This must be inlined in a function which
   uses global data.  */

static inline Elf64_Addr
elf_machine_dynamic (void)
{
  register Elf64_Addr *got;

  __asm__ ( "	larl   %0,_GLOBAL_OFFSET_TABLE_\n"
	    : "=&a" (got) : : "0" );

  return *got;
}

/* Return the run-time load address of the shared object.  */
static inline Elf64_Addr
elf_machine_load_address (void)
{
  Elf64_Addr addr;

  __asm__( "   larl	 %0,_dl_start\n"
	   "   larl	 1,_GLOBAL_OFFSET_TABLE_\n"
	   "   lghi	 2,_dl_start@GOT\n"
	   "   slg	 %0,0(2,1)"
	   : "=&d" (addr) : : "1", "2" );
  return addr;
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((unused))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  extern void _dl_runtime_resolve (Elf64_Word);
  extern void _dl_runtime_profile (Elf64_Word);
#if defined HAVE_S390_VX_ASM_SUPPORT
  extern void _dl_runtime_resolve_vx (Elf64_Word);
  extern void _dl_runtime_profile_vx (Elf64_Word);
#endif

  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* The GOT entries for functions in the PLT have not yet been filled
	 in.  Their initial contents will arrange when called to push an
	 offset into the .rela.plt section, push _GLOBAL_OFFSET_TABLE_[1],
	 and then jump to _GLOBAL_OFFSET_TABLE[2].  */
      Elf64_Addr *got;
      got = (Elf64_Addr *) D_PTR (l, l_info[DT_PLTGOT]);
      /* If a library is prelinked but we have to relocate anyway,
	 we have to be able to undo the prelinking of .got.plt.
	 The prelinker saved us here address of .plt + 0x2e.  */
      if (got[1])
	{
	  l->l_mach.plt = got[1] + l->l_addr;
	  l->l_mach.jmprel = (const Elf64_Rela *) D_PTR (l, l_info[DT_JMPREL]);
	}
      got[1] = (Elf64_Addr) l;	/* Identify this shared object.	 */

      /* The got[2] entry contains the address of a function which gets
	 called to get the address of a so far unresolved function and
	 jump to it.  The profiling extension of the dynamic linker allows
	 to intercept the calls to collect information.	 In this case we
	 don't store the address in the GOT so that all future calls also
	 end in this function.	*/
      if (__glibc_unlikely (profile))
	{
#if defined HAVE_S390_VX_ASM_SUPPORT
	  if (GLRO(dl_hwcap) & HWCAP_S390_VX)
	    got[2] = (Elf64_Addr) &_dl_runtime_profile_vx;
	  else
	    got[2] = (Elf64_Addr) &_dl_runtime_profile;
#else
	  got[2] = (Elf64_Addr) &_dl_runtime_profile;
#endif

	  if (GLRO(dl_profile) != NULL
	      && _dl_name_match_p (GLRO(dl_profile), l))
	    /* This is the object we are looking for.  Say that we really
	       want profiling and the timers are started.  */
	    GL(dl_profile_map) = l;
	}
      else
	{
	  /* This function will get called to fix up the GOT entry indicated by
	     the offset on the stack, and then jump to the resolved address.  */
#if defined HAVE_S390_VX_ASM_SUPPORT
	  if (GLRO(dl_hwcap) & HWCAP_S390_VX)
	    got[2] = (Elf64_Addr) &_dl_runtime_resolve_vx;
	  else
	    got[2] = (Elf64_Addr) &_dl_runtime_resolve;
#else
	  got[2] = (Elf64_Addr) &_dl_runtime_resolve;
#endif
	}
    }

  return lazy;
}

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.	*/

#define RTLD_START __asm__ ("\n\
.text\n\
.align 4\n\
.globl _start\n\
.globl _dl_start_user\n\
_start:\n\
	lgr   %r2,%r15\n\
	# Alloc stack frame\n\
	aghi  %r15,-160\n\
	# Set the back chain to zero\n\
	xc    0(8,%r15),0(%r15)\n\
	# Call _dl_start with %r2 pointing to arg on stack\n\
	brasl %r14,_dl_start	     # call _dl_start\n\
_dl_start_user:\n\
	# Save the user entry point address in %r8.\n\
	lgr   %r8,%r2\n\
	# Point %r12 at the GOT.\n\
	larl  %r12,_GLOBAL_OFFSET_TABLE_\n\
	# See if we were run as a command with the executable file\n\
	# name as an extra leading argument.\n\
	lghi  %r1,_dl_skip_args@GOT\n\
	lg    %r1,0(%r1,%r12)\n\
	lgf   %r1,0(%r1)	  # load _dl_skip_args\n\
	# Get the original argument count.\n\
	lg    %r0,160(%r15)\n\
	# Subtract _dl_skip_args from it.\n\
	sgr   %r0,%r1\n\
	# Adjust the stack pointer to skip _dl_skip_args words.\n\
	sllg  %r1,%r1,3\n\
	agr   %r15,%r1\n\
	# Set the back chain to zero again\n\
	xc    0(8,%r15),0(%r15)\n\
	# Store back the modified argument count.\n\
	stg   %r0,160(%r15)\n\
	# The special initializer gets called with the stack just\n\
	# as the application's entry point will see it; it can\n\
	# switch stacks if it moves these contents over.\n\
" RTLD_START_SPECIAL_INIT "\n\
	# Call the function to run the initializers.\n\
	# Load the parameters:\n\
	# (%r2, %r3, %r4, %r5) = (_dl_loaded, argc, argv, envp)\n\
	lghi  %r2,_rtld_local@GOT\n\
	lg    %r2,0(%r2,%r12)\n\
	lg    %r2,0(%r2)\n\
	lg    %r3,160(%r15)\n\
	la    %r4,168(%r15)\n\
	lgr   %r5,%r3\n\
	sllg  %r5,%r5,3\n\
	la    %r5,176(%r5,%r15)\n\
	brasl %r14,_dl_init@PLT\n\
	# Pass our finalizer function to the user in %r14, as per ELF ABI.\n\
	lghi  %r14,_dl_fini@GOT\n\
	lg    %r14,0(%r14,%r12)\n\
	# Free stack frame\n\
	aghi  %r15,160\n\
	# Jump to the user's entry point (saved in %r8).\n\
	br    %r8\n\
");

#ifndef RTLD_START_SPECIAL_INIT
#define RTLD_START_SPECIAL_INIT /* nothing */
#endif

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry or
   TLS variable, so undefined references should not be allowed to
   define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#define elf_machine_type_class(type) \
  ((((type) == R_390_JMP_SLOT || (type) == R_390_TLS_DTPMOD		      \
     || (type) == R_390_TLS_DTPOFF || (type) == R_390_TLS_TPOFF)	      \
    * ELF_RTYPE_CLASS_PLT)						      \
   | (((type) == R_390_COPY) * ELF_RTYPE_CLASS_COPY))

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	R_390_JMP_SLOT

/* The 64 bit S/390 never uses Elf64_Rel relocations.  */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

/* We define an initialization functions.  This is called very early in
   _dl_sysdep_start.  */
#define DL_PLATFORM_INIT dl_platform_init ()

static inline void __attribute__ ((unused))
dl_platform_init (void)
{
  if (GLRO(dl_platform) != NULL && *GLRO(dl_platform) == '\0')
    /* Avoid an empty string which would disturb us.  */
    GLRO(dl_platform) = NULL;
}

static inline Elf64_Addr
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf64_Rela *reloc,
		       Elf64_Addr *reloc_addr, Elf64_Addr value)
{
  return *reloc_addr = value;
}

/* Return the final value of a plt relocation.	*/
static inline Elf64_Addr
elf_machine_plt_value (struct link_map *map, const Elf64_Rela *reloc,
		       Elf64_Addr value)
{
  return value;
}

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER s390_64_gnu_pltenter
#define ARCH_LA_PLTEXIT s390_64_gnu_pltexit

#endif /* !dl_machine_h */

#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

auto inline void
__attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const Elf64_Rela *reloc,
		  const Elf64_Sym *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf64_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF64_R_TYPE (reloc->r_info);

#if !defined RTLD_BOOTSTRAP || !defined HAVE_Z_COMBRELOC
  if (__glibc_unlikely (r_type == R_390_RELATIVE))
    {
# if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC
      /* This is defined in rtld.c, but nowhere in the static libc.a;
	 make the reference weak so static programs can still link.
	 This declaration cannot be done when compiling rtld.c
	 (i.e. #ifdef RTLD_BOOTSTRAP) because rtld.c contains the
	 common defn for _dl_rtld_map, which is incompatible with a
	 weak decl in the same file.  */
#  ifndef SHARED
      weak_extern (GL(dl_rtld_map));
#  endif
      if (map != &GL(dl_rtld_map)) /* Already done in rtld itself.  */
# endif
	*reloc_addr = map->l_addr + reloc->r_addend;
    }
  else
#endif
  if (__glibc_unlikely (r_type == R_390_NONE))
    return;
  else
    {
#if !defined RTLD_BOOTSTRAP && !defined RESOLVE_CONFLICT_FIND_MAP
      /* Only needed for R_390_COPY below.  */
      const Elf64_Sym *const refsym = sym;
#endif
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf64_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      if (sym != NULL
	  && __builtin_expect (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC,
			       0)
	  && __builtin_expect (sym->st_shndx != SHN_UNDEF, 1)
	  && __builtin_expect (!skip_ifunc, 1))
	value = elf_ifunc_invoke (value);

      switch (r_type)
	{
	case R_390_IRELATIVE:
	  value = map->l_addr + reloc->r_addend;
	  if (__glibc_likely (!skip_ifunc))
	    value = elf_ifunc_invoke (value);
	  *reloc_addr = value;
	  break;
	case R_390_GLOB_DAT:
	case R_390_JMP_SLOT:
	  *reloc_addr = value + reloc->r_addend;
	  break;

#ifndef RESOLVE_CONFLICT_FIND_MAP
	case R_390_TLS_DTPMOD:
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
	case R_390_TLS_DTPOFF:
# ifndef RTLD_BOOTSTRAP
	  /* During relocation all TLS symbols are defined and used.
	     Therefore the offset is already correct.  */
	  if (sym != NULL)
	    *reloc_addr = sym->st_value + reloc->r_addend;
# endif
	  break;
	case R_390_TLS_TPOFF:
	  /* The offset is negative, forward from the thread pointer.  */
# ifdef RTLD_BOOTSTRAP
	  *reloc_addr = sym->st_value + reloc->r_addend - map->l_tls_offset;
# else
	  /* We know the offset of the object the symbol is contained in.
	     It is a negative value which will be added to the
	     thread pointer.  */
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = (sym->st_value + reloc->r_addend
			     - sym_map->l_tls_offset);
	    }
#endif
	  break;
#endif  /* use TLS */

#ifndef RTLD_BOOTSTRAP
# ifndef RESOLVE_CONFLICT_FIND_MAP
	/* Not needed for dl-conflict.c.  */
	case R_390_COPY:
	  if (sym == NULL)
	    /* This can happen in trace mode if an object could not be
	       found.  */
	    break;
	  if (__builtin_expect (sym->st_size > refsym->st_size, 0)
	      || (__builtin_expect (sym->st_size < refsym->st_size, 0)
		  && __builtin_expect (GLRO(dl_verbose), 0)))
	    {
	      const char *strtab;

	      strtab = (const char *) D_PTR (map,l_info[DT_STRTAB]);
	      _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				RTLD_PROGNAME, strtab + refsym->st_name);
	    }
	  memcpy (reloc_addr_arg, (void *) value,
		  MIN (sym->st_size, refsym->st_size));
	  break;
# endif
	case R_390_64:
	  *reloc_addr = value + reloc->r_addend;
	  break;
	case R_390_32:
	  *(unsigned int *) reloc_addr = value + reloc->r_addend;
	  break;
	case R_390_16:
	  *(unsigned short *) reloc_addr = value + reloc->r_addend;
	  break;
	case R_390_8:
	  *(char *) reloc_addr = value + reloc->r_addend;
	  break;
# ifndef RESOLVE_CONFLICT_FIND_MAP
	case R_390_PC64:
	  *reloc_addr = value +reloc->r_addend - (Elf64_Addr) reloc_addr;
	  break;
	case R_390_PC32DBL:
	  *(unsigned int *) reloc_addr = (unsigned int)
	    ((int) (value + reloc->r_addend - (Elf64_Addr) reloc_addr) >> 1);
	  break;
	case R_390_PC32:
	  *(unsigned int *) reloc_addr =
	    value + reloc->r_addend - (Elf64_Addr) reloc_addr;
	  break;
	case R_390_PC16DBL:
	  *(unsigned short *) reloc_addr = (unsigned short)
	    ((short) (value + reloc->r_addend - (Elf64_Addr) reloc_addr) >> 1);
	  break;
	case R_390_PC16:
	  *(unsigned short *) reloc_addr =
	    value + reloc->r_addend - (Elf64_Addr) reloc_addr;
	  break;
	case R_390_NONE:
	  break;
# endif
#endif
#if !defined(RTLD_BOOTSTRAP) || defined(_NDEBUG)
	default:
	  /* We add these checks in the version to relocate ld.so only
	     if we are still debugging.	 */
	  _dl_reloc_bad_type (map, r_type, 0);
	  break;
#endif
	}
    }
}

auto inline void
__attribute__ ((always_inline))
elf_machine_rela_relative (Elf64_Addr l_addr, const Elf64_Rela *reloc,
			   void *const reloc_addr_arg)
{
  Elf64_Addr *const reloc_addr = reloc_addr_arg;
  *reloc_addr = l_addr + reloc->r_addend;
}

auto inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf64_Addr l_addr, const Elf64_Rela *reloc,
		      int skip_ifunc)
{
  Elf64_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELF64_R_TYPE (reloc->r_info);
  /* Check for unexpected PLT reloc type.  */
  if (__glibc_likely (r_type == R_390_JMP_SLOT))
    {
      if (__builtin_expect (map->l_mach.plt, 0) == 0)
	*reloc_addr += l_addr;
      else
	*reloc_addr = map->l_mach.plt + (reloc - map->l_mach.jmprel) * 32;
    }
  else if (__glibc_likely (r_type == R_390_IRELATIVE))
    {
      Elf64_Addr value = map->l_addr + reloc->r_addend;
      if (__glibc_likely (!skip_ifunc))
	value = elf_ifunc_invoke (value);
      *reloc_addr = value;
    }
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

#endif /* RESOLVE_MAP */
