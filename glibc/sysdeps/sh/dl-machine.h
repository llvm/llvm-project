/* Machine-dependent ELF dynamic relocation inline functions.  SH version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#define ELF_MACHINE_NAME "SH"

#include <sys/param.h>
#include <sysdep.h>
#include <assert.h>

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int __attribute__ ((unused))
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_SH;
}


/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  This must be inlined in a function which
   uses global data.  */
static inline Elf32_Addr __attribute__ ((unused))
elf_machine_dynamic (void)
{
  register Elf32_Addr *got;
  asm ("mov r12,%0" :"=r" (got));
  return *got;
}


/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr __attribute__ ((unused))
elf_machine_load_address (void)
{
  Elf32_Addr addr;
  asm ("mov.l 1f,r0\n\
	mov.l 3f,r2\n\
	add r12,r2\n\
	mov.l @(r0,r12),r0\n\
	bra 2f\n\
	 sub r0,r2\n\
	.align 2\n\
	1: .long _dl_start@GOT\n\
	3: .long _dl_start@GOTOFF\n\
	2: mov r2,%0"
       : "=r" (addr) : : "r0", "r1", "r2");
  return addr;
}


/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((unused, always_inline))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  Elf32_Addr *got;
  extern void _dl_runtime_resolve (Elf32_Word);
  extern void _dl_runtime_profile (Elf32_Word);

  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* The GOT entries for functions in the PLT have not yet been filled
	 in.  Their initial contents will arrange when called to load an
	 offset into the .rela.plt section and _GLOBAL_OFFSET_TABLE_[1],
	 and then jump to _GLOBAL_OFFSET_TABLE[2].  */
      got = (Elf32_Addr *) D_PTR (l, l_info[DT_PLTGOT]);
      /* If a library is prelinked but we have to relocate anyway,
	 we have to be able to undo the prelinking of .got.plt.
	 The prelinker saved us here address of .plt + 36.  */
      if (got[1])
	{
	  l->l_mach.plt = got[1] + l->l_addr;
	  l->l_mach.gotplt = (Elf32_Addr) &got[3];
	}
      got[1] = (Elf32_Addr) l;	/* Identify this shared object.	 */

      /* The got[2] entry contains the address of a function which gets
	 called to get the address of a so far unresolved function and
	 jump to it.  The profiling extension of the dynamic linker allows
	 to intercept the calls to collect information.	 In this case we
	 don't store the address in the GOT so that all future calls also
	 end in this function.	*/
      if (profile)
	{
	  got[2] = (Elf32_Addr) &_dl_runtime_profile;
	  /* Say that we really want profiling and the timers are started.  */
	  if (GLRO(dl_profile) != NULL
	      && _dl_name_match_p (GLRO(dl_profile), l))
	    GL(dl_profile_map) = l;
	}
      else
	/* This function will get called to fix up the GOT entry indicated by
	   the offset on the stack, and then jump to the resolved address.  */
	got[2] = (Elf32_Addr) &_dl_runtime_resolve;
    }
  return lazy;
}

#define ELF_MACHINE_RUNTIME_FIXUP_ARGS int plt_type
#define ELF_MACHINE_RUNTIME_FIXUP_PARAMS plt_type

/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK	0x80000000UL

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.	*/

#define RTLD_START asm ("\
.text\n\
.globl _start\n\
.globl _dl_start_user\n\
_start:\n\
	mov r15,r4\n\
	mov.l .L_dl_start,r1\n\
	mova .L_dl_start,r0\n\
	add r1,r0\n\
	jsr @r0\n\
	 nop\n\
_dl_start_user:\n\
	! Save the user entry point address in r8.\n\
	mov r0,r8\n\
	! Point r12 at the GOT.\n\
	mov.l 1f,r12\n\
	mova 1f,r0\n\
	bra 2f\n\
	 add r0,r12\n\
	.align 2\n\
1:	.long _GLOBAL_OFFSET_TABLE_\n\
2:	! See if we were run as a command with the executable file\n\
	! name as an extra leading argument.\n\
	mov.l .L_dl_skip_args,r0\n\
	mov.l @(r0,r12),r0\n\
	mov.l @r0,r0\n\
	! Get the original argument count.\n\
	mov.l @r15,r5\n\
	! Subtract _dl_skip_args from it.\n\
	sub r0,r5\n\
	! Adjust the stack pointer to skip _dl_skip_args words.\n\
	shll2 r0\n\
	add r0,r15\n\
	! Store back the modified argument count.\n\
	mov.l r5,@r15\n\
	! Compute argv address and envp.\n\
	mov r15,r6\n\
	add #4,r6\n\
	mov r5,r7\n\
	shll2 r7\n\
	add r15,r7\n\
	add #8,r7\n\
	mov.l .L_dl_loaded,r0\n\
	mov.l @(r0,r12),r0\n\
	mov.l @r0,r4\n\
	! Call _dl_init.\n\
	mov.l .L_dl_init,r1\n\
	mova .L_dl_init,r0\n\
	add r1,r0\n\
	jsr @r0\n\
	 nop\n\
1:	! Pass our finalizer function to the user in r4, as per ELF ABI.\n\
	mov.l .L_dl_fini,r0\n\
	mov.l @(r0,r12),r4\n\
	! Jump to the user's entry point.\n\
	jmp @r8\n\
	 nop\n\
	.align 2\n\
.L_dl_start:\n\
	.long _dl_start@PLT\n\
.L_dl_skip_args:\n\
	.long _dl_skip_args@GOT\n\
.L_dl_init:\n\
	.long _dl_init@PLT\n\
.L_dl_loaded:\n\
	.long _rtld_local@GOT\n\
.L_dl_fini:\n\
	.long _dl_fini@GOT\n\
	.type __fpscr_values,@object\n\
	.global __fpscr_values\n\
__fpscr_values:\n\
	.long   0\n\
	.long   0x80000\n\
	.weak __fpscr_values\n\
.previous\n\
");

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry or
   TLS variable, so undefined references should not be allowed to
   define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#define elf_machine_type_class(type) \
  ((((type) == R_SH_JMP_SLOT || (type) == R_SH_TLS_DTPMOD32		      \
     || (type) == R_SH_TLS_DTPOFF32 || (type) == R_SH_TLS_TPOFF32)	      \
    * ELF_RTYPE_CLASS_PLT)						      \
   | (((type) == R_SH_COPY) * ELF_RTYPE_CLASS_COPY))

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	R_SH_JMP_SLOT

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

/* Return the final value of a plt relocation.	*/
static inline Elf32_Addr
elf_machine_plt_value (struct link_map *map, const Elf32_Rela *reloc,
		       Elf32_Addr value)
{
  return value + reloc->r_addend;
}

#define ARCH_LA_PLTENTER sh_gnu_pltenter
#define ARCH_LA_PLTEXIT sh_gnu_pltexit

#endif /* !dl_machine_h */

/* SH never uses Elf32_Rel relocations.	 */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

auto inline void
__attribute ((always_inline))
elf_machine_rela (struct link_map *map, const Elf32_Rela *reloc,
		  const Elf32_Sym *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);
  Elf32_Addr value;

#define COPY_UNALIGNED_WORD(swp, twp, align) \
  { \
    void *__s = (swp), *__t = (twp); \
    unsigned char *__s1 = __s, *__t1 = __t; \
    unsigned short *__s2 = __s, *__t2 = __t; \
    unsigned long *__s4 = __s, *__t4 = __t; \
    switch ((align)) \
    { \
    case 0: \
      *__t4 = *__s4; \
      break; \
    case 2: \
      *__t2++ = *__s2++; \
      *__t2 = *__s2; \
      break; \
    default: \
      *__t1++ = *__s1++; \
      *__t1++ = *__s1++; \
      *__t1++ = *__s1++; \
      *__t1 = *__s1; \
      break; \
    } \
  }

  if (__glibc_unlikely (r_type == R_SH_RELATIVE))
    {
#ifndef RTLD_BOOTSTRAP
      if (map != &GL(dl_rtld_map)) /* Already done in rtld itself.	 */
#endif
	{
	  if (reloc->r_addend)
	    value = map->l_addr + reloc->r_addend;
	  else
	    {
	      COPY_UNALIGNED_WORD (reloc_addr_arg, &value,
				   (int) reloc_addr_arg & 3);
	      value += map->l_addr;
	    }
	  COPY_UNALIGNED_WORD (&value, reloc_addr_arg,
			       (int) reloc_addr_arg & 3);
	}
    }
#ifndef RTLD_BOOTSTRAP
  else if (__glibc_unlikely (r_type == R_SH_NONE))
    return;
#endif
  else
    {
      const Elf32_Sym *const refsym = sym;
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);

      value = SYMBOL_ADDRESS (sym_map, sym, true);
      value += reloc->r_addend;

      switch (r_type)
	{
	case R_SH_COPY:
	  if (sym == NULL)
	    /* This can happen in trace mode if an object could not be
	       found.  */
	    break;
	  if (sym->st_size > refsym->st_size
	      || (sym->st_size < refsym->st_size && GLRO(dl_verbose)))
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
	case R_SH_GLOB_DAT:
	case R_SH_JMP_SLOT:
	  /* These addresses are always aligned.  */
	  *reloc_addr = value;
	  break;
	  /* XXX Remove TLS relocations which are not needed.  */
	case R_SH_TLS_DTPMOD32:
#ifdef RTLD_BOOTSTRAP
	  /* During startup the dynamic linker is always the module
	     with index 1.
	     XXX If this relocation is necessary move before RESOLVE
	     call.  */
	  *reloc_addr = 1;
#else
	  /* Get the information from the link map returned by the
	     resolv function.  */
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
#endif
	  break;
	case R_SH_TLS_DTPOFF32:
#ifndef RTLD_BOOTSTRAP
	  /* During relocation all TLS symbols are defined and used.
	     Therefore the offset is already correct.  */
	  if (sym != NULL)
	    *reloc_addr = sym->st_value;
#endif
	  break;
	case R_SH_TLS_TPOFF32:
	  /* The offset is positive, afterward from the thread pointer.  */
#ifdef RTLD_BOOTSTRAP
	  *reloc_addr = map->l_tls_offset + sym->st_value + reloc->r_addend;
#else
	  /* We know the offset of object the symbol is contained in.
	     It is a positive value which will be added to the thread
	     pointer.  To get the variable position in the TLS block
	     we add the offset from that of the TLS block.  */
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = sym_map->l_tls_offset + sym->st_value
			    + reloc->r_addend;
	    }
#endif
	  break;
	case R_SH_DIR32:
	  {
#if !defined RTLD_BOOTSTRAP && !defined RESOLVE_CONFLICT_FIND_MAP
	   /* This is defined in rtld.c, but nowhere in the static
	      libc.a; make the reference weak so static programs can
	      still link.  This declaration cannot be done when
	      compiling rtld.c (i.e. #ifdef RTLD_BOOTSTRAP) because
	      rtld.c contains the common defn for _dl_rtld_map, which
	      is incompatible with a weak decl in the same file.  */
# ifndef SHARED
	    weak_extern (_dl_rtld_map);
# endif
	    if (map == &GL(dl_rtld_map))
	      /* Undo the relocation done here during bootstrapping.
		 Now we will relocate it anew, possibly using a
		 binding found in the user program or a loaded library
		 rather than the dynamic linker's built-in definitions
		 used while loading those libraries.  */
	      value -= SYMBOL_ADDRESS (map, refsym, true) + reloc->r_addend;
#endif
	    COPY_UNALIGNED_WORD (&value, reloc_addr_arg,
				 (int) reloc_addr_arg & 3);
	    break;
	  }
	case R_SH_REL32:
	  value = (value - (Elf32_Addr) reloc_addr);
	  COPY_UNALIGNED_WORD (&value, reloc_addr_arg,
			       (int) reloc_addr_arg & 3);
	  break;
	default:
	  _dl_reloc_bad_type (map, r_type, 0);
	  break;
	}
    }
}

auto inline void
__attribute__ ((always_inline))
elf_machine_rela_relative (Elf32_Addr l_addr, const Elf32_Rela *reloc,
			   void *const reloc_addr_arg)
{
  Elf32_Addr value;

  if (reloc->r_addend)
    value = l_addr + reloc->r_addend;
  else
    {
      COPY_UNALIGNED_WORD (reloc_addr_arg, &value, (int) reloc_addr_arg & 3);
      value += l_addr;
    }
  COPY_UNALIGNED_WORD (&value, reloc_addr_arg, (int) reloc_addr_arg & 3);

#undef COPY_UNALIGNED_WORD
}

auto inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rela *reloc,
		      int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  /* Check for unexpected PLT reloc type.  */
  if (ELF32_R_TYPE (reloc->r_info) == R_SH_JMP_SLOT)
    {
      if (__builtin_expect (map->l_mach.plt, 0) == 0)
	*reloc_addr += l_addr;
      else
	*reloc_addr =
	  map->l_mach.plt
	  + (((Elf32_Addr) reloc_addr) - map->l_mach.gotplt) * 7;
    }
  else
    _dl_reloc_bad_type (map, ELF32_R_TYPE (reloc->r_info), 1);
}

#endif /* RESOLVE_MAP */
