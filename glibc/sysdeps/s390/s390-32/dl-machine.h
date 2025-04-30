/* Machine-dependent ELF dynamic relocation inline functions.  S390 Version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Carl Pederson & Martin Schwidefsky.
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

#define ELF_MACHINE_NAME "s390"

#include <sys/param.h>
#include <string.h>
#include <link.h>
#include <sysdeps/s390/dl-procinfo.h>
#include <dl-irel.h>

/* This is an older, now obsolete value.  */
#define EM_S390_OLD	0xA390

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  /* Check if the kernel provides the high gpr facility if needed by
     the binary.  */
  if ((ehdr->e_flags & EF_S390_HIGH_GPRS)
      && !(GLRO (dl_hwcap) & HWCAP_S390_HIGH_GPRS))
    return 0;

  return (ehdr->e_machine == EM_S390 || ehdr->e_machine == EM_S390_OLD)
	 && ehdr->e_ident[EI_CLASS] == ELFCLASS32;
}


/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  This must be inlined in a function which
   uses global data.  */

static inline Elf32_Addr
elf_machine_dynamic (void)
{
  register Elf32_Addr *got;

  __asm__( "        bras   %0,2f\n"
	   "1:      .long  _GLOBAL_OFFSET_TABLE_-1b\n"
	   "2:      al     %0,0(%0)"
	   : "=&a" (got) : : "0" );

  return *got;
}


/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr
elf_machine_load_address (void)
{
  Elf32_Addr addr;

  __asm__( "   bras  1,2f\n"
	   "1: .long _GLOBAL_OFFSET_TABLE_ - 1b\n"
	   "   .long (_dl_start - 1b - 0x80000000) & 0x00000000ffffffff\n"
	   "2: l     %0,4(1)\n"
	   "   ar    %0,1\n"
	   "   al    1,0(1)\n"
	   "   sl    %0,_dl_start@GOT(1)"
	   : "=&d" (addr) : : "1" );
  return addr;
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((unused))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  extern void _dl_runtime_resolve (Elf32_Word);
  extern void _dl_runtime_profile (Elf32_Word);
#if defined HAVE_S390_VX_ASM_SUPPORT
  extern void _dl_runtime_resolve_vx (Elf32_Word);
  extern void _dl_runtime_profile_vx (Elf32_Word);
#endif


  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* The GOT entries for functions in the PLT have not yet been filled
	 in.  Their initial contents will arrange when called to push an
	 offset into the .rel.plt section, push _GLOBAL_OFFSET_TABLE_[1],
	 and then jump to _GLOBAL_OFFSET_TABLE[2].  */
      Elf32_Addr *got;
      got = (Elf32_Addr *) D_PTR (l, l_info[DT_PLTGOT]);
      /* If a library is prelinked but we have to relocate anyway,
	 we have to be able to undo the prelinking of .got.plt.
	 The prelinker saved us here address of .plt + 0x2c.  */
      if (got[1])
	{
	  l->l_mach.plt = got[1] + l->l_addr;
	  l->l_mach.jmprel = (const Elf32_Rela *) D_PTR (l, l_info[DT_JMPREL]);
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
#if defined HAVE_S390_VX_ASM_SUPPORT
	  if (GLRO(dl_hwcap) & HWCAP_S390_VX)
	    got[2] = (Elf32_Addr) &_dl_runtime_profile_vx;
	  else
	    got[2] = (Elf32_Addr) &_dl_runtime_profile;
#else
	  got[2] = (Elf32_Addr) &_dl_runtime_profile;
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
	    got[2] = (Elf32_Addr) &_dl_runtime_resolve_vx;
	  else
	    got[2] = (Elf32_Addr) &_dl_runtime_resolve;
#else
	  got[2] = (Elf32_Addr) &_dl_runtime_resolve;
#endif
	}
    }

  return lazy;
}

/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK   0xf8000000UL

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_START __asm__ ("\n\
.text\n\
.align 4\n\
.globl _start\n\
.globl _dl_start_user\n\
_start:\n\
	basr  %r13,0\n\
0:      ahi   %r13,.Llit-0b\n\
	lr    %r2,%r15\n\
	# Alloc stack frame\n\
	ahi   %r15,-96\n\
	# Set the back chain to zero\n\
	xc    0(4,%r15),0(%r15)\n\
	# Call _dl_start with %r2 pointing to arg on stack\n\
	l     %r14,.Ladr1-.Llit(%r13)\n\
	bas   %r14,0(%r14,%r13)   # call _dl_start\n\
_dl_start_user:\n\
	# Save the user entry point address in %r8.\n\
	lr    %r8,%r2\n\
	# Point %r12 at the GOT.\n\
	l     %r12,.Ladr0-.Llit(%r13)\n\
	ar    %r12,%r13\n\
	# See if we were run as a command with the executable file\n\
	# name as an extra leading argument.\n\
	l     %r1,_dl_skip_args@GOT(%r12)\n\
	l     %r1,0(%r1)	# load _dl_skip_args\n\
	ltr   %r1,%r1\n\
	je    4f		# Skip the arg adjustment if there were none.\n\
	# Get the original argument count.\n\
	l     %r0,96(%r15)\n\
	# Subtract _dl_skip_args from it.\n\
	sr    %r0,%r1\n\
	# Store back the modified argument count.\n\
	st    %r0,96(%r15)\n\
	# Copy argv and envp forward to account for skipped argv entries.\n\
	# We skipped at least one argument or we would not get here.\n\
	la    %r6,100(%r15)	# Destination pointer i.e. &argv[0]\n\
	lr    %r5,%r6\n\
	lr    %r0,%r1\n\
	sll   %r0,2\n		# Number of skipped bytes.\n\
	ar    %r5,%r0		# Source pointer = Dest + Skipped args.\n\
	# argv copy loop:\n\
1:	l     %r7,0(%r5)	# Load a word from the source.\n\
	st    %r7,0(%r6)	# Store the word in the destination.\n\
	ahi   %r5,4\n\
	ahi   %r6,4\n\
	ltr   %r7,%r7\n\
	jne   1b		# Stop after copying the NULL.\n\
	# envp copy loop:\n\
2:	l     %r7,0(%r5)	# Load a word from the source.\n\
	st    %r7,0(%r6)	# Store the word in the destination.\n\
	ahi   %r5,4\n\
	ahi   %r6,4\n\
	ltr   %r7,%r7\n\
	jne   2b		# Stop after copying the NULL.\n\
	# Now we have to zero out the envp entries after NULL to allow\n\
	# start.S to properly find auxv by skipping zeroes.\n\
	# zero out loop:\n\
	lhi   %r7,0\n\
3:	st    %r7,0(%r6)	# Store zero.\n\
	ahi   %r6,4		# Advance dest pointer.\n\
	ahi   %r1,-1		# Subtract one from the word count.\n\
	ltr   %r1,%r1\n\
	jne    3b		# Keep copying if the word count is non-zero.\n\
	# Adjust _dl_argv\n\
	la    %r6,100(%r15)\n\
	l     %r1,_dl_argv@GOT(%r12)\n\
	st    %r6,0(%r1)\n\
	# The special initializer gets called with the stack just\n\
	# as the application's entry point will see it; it can\n\
	# switch stacks if it moves these contents over.\n\
" RTLD_START_SPECIAL_INIT "\n\
	# Call the function to run the initializers.\n\
	# Load the parameters:\n\
	# (%r2, %r3, %r4, %r5) = (_dl_loaded, argc, argv, envp)\n\
4:	l     %r2,_rtld_local@GOT(%r12)\n\
	l     %r2,0(%r2)\n\
	l     %r3,96(%r15)\n\
	la    %r4,100(%r15)\n\
	lr    %r5,%r3\n\
	sll   %r5,2\n\
	la    %r5,104(%r5,%r15)\n\
	l     %r1,.Ladr4-.Llit(%r13)\n\
	bas   %r14,0(%r1,%r13)\n\
	# Pass our finalizer function to the user in %r14, as per ELF ABI.\n\
	l     %r14,_dl_fini@GOT(%r12)\n\
	# Free stack frame\n\
	ahi   %r15,96\n\
	# Jump to the user's entry point (saved in %r8).\n\
	br    %r8\n\
.Llit:\n\
.Ladr0: .long _GLOBAL_OFFSET_TABLE_-.Llit\n\
.Ladr1: .long _dl_start-.Llit\n\
.Ladr4: .long _dl_init@PLT-.Llit\n\
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
#define ELF_MACHINE_JMP_SLOT    R_390_JMP_SLOT

/* The S390 never uses Elf32_Rel relocations.  */
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

static inline Elf32_Addr
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf32_Rela *reloc,
		       Elf32_Addr *reloc_addr, Elf32_Addr value)
{
  return *reloc_addr = value;
}

/* Return the final value of a plt relocation.  */
static inline Elf32_Addr
elf_machine_plt_value (struct link_map *map, const Elf32_Rela *reloc,
		       Elf32_Addr value)
{
  return value;
}

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER s390_32_gnu_pltenter
#define ARCH_LA_PLTEXIT s390_32_gnu_pltexit

#endif /* !dl_machine_h */


#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

auto inline void
__attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const Elf32_Rela *reloc,
		  const Elf32_Sym *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);

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
      const Elf32_Sym *const refsym = sym;
#endif
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf32_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      if (sym != NULL
	  && __builtin_expect (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC, 0)
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
	/* Not needed in dl-conflict.c.  */
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

	      strtab = (const char *) D_PTR(map,l_info[DT_STRTAB]);
	      _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				RTLD_PROGNAME, strtab + refsym->st_name);
	    }
	  memcpy (reloc_addr_arg, (void *) value,
		  MIN (sym->st_size, refsym->st_size));
	  break;
# endif
	case R_390_32:
	  *reloc_addr = value + reloc->r_addend;
	  break;
	case R_390_16:
	  *(unsigned short *) reloc_addr = value + reloc->r_addend;
	  break;
	case R_390_8:
	  *(char *) reloc_addr = value + reloc->r_addend;
	  break;
# ifndef RESOLVE_CONFLICT_FIND_MAP
	case R_390_PC32:
	  *reloc_addr = value + reloc->r_addend - (Elf32_Addr) reloc_addr;
	  break;
	case R_390_PC16DBL:
	  *(unsigned short *) reloc_addr = (unsigned short)
	    ((short) (value + reloc->r_addend - (Elf32_Addr) reloc_addr) >> 1);
	  break;
	case R_390_PC32DBL:
	  *(unsigned int *) reloc_addr = (unsigned int)
	    ((int) (value + reloc->r_addend - (Elf32_Addr) reloc_addr) >> 1);
	  break;
	case R_390_PC16:
	  *(unsigned short *) reloc_addr =
	    value + reloc->r_addend - (Elf32_Addr) reloc_addr;
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
elf_machine_rela_relative (Elf32_Addr l_addr, const Elf32_Rela *reloc,
			   void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  *reloc_addr = l_addr + reloc->r_addend;
}

auto inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rela *reloc,
		      int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);
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
      Elf32_Addr value = map->l_addr + reloc->r_addend;
      if (__glibc_likely (!skip_ifunc))
	value = elf_ifunc_invoke (value);
      *reloc_addr = value;
    }
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

#endif /* RESOLVE_MAP */
