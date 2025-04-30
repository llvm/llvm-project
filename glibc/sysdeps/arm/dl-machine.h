/* Machine-dependent ELF dynamic relocation inline functions.  ARM version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef dl_machine_h
#define dl_machine_h

#define ELF_MACHINE_NAME "ARM"

#include <sys/param.h>
#include <tls.h>
#include <dl-tlsdesc.h>
#include <dl-irel.h>

#ifndef CLEAR_CACHE
# error CLEAR_CACHE definition required to handle TEXTREL
#endif

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int __attribute__ ((unused))
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_ARM;
}


/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  */
static inline Elf32_Addr __attribute__ ((unused))
elf_machine_dynamic (void)
{
  /* Declaring this hidden ensures that a PC-relative reference is used.  */
  extern const Elf32_Addr _GLOBAL_OFFSET_TABLE_[] attribute_hidden;
  return _GLOBAL_OFFSET_TABLE_[0];
}


/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr __attribute__ ((unused))
elf_machine_load_address (void)
{
  Elf32_Addr pcrel_addr;
#ifdef SHARED
  extern Elf32_Addr __dl_start (void *) asm ("_dl_start");
  Elf32_Addr got_addr = (Elf32_Addr) &__dl_start;
  asm ("adr %0, _dl_start" : "=r" (pcrel_addr));
#else
  extern Elf32_Addr __dl_relocate_static_pie (void *)
    asm ("_dl_relocate_static_pie") attribute_hidden;
  Elf32_Addr got_addr = (Elf32_Addr) &__dl_relocate_static_pie;
  asm ("adr %0, _dl_relocate_static_pie" : "=r" (pcrel_addr));
#endif
#ifdef __thumb__
  /* Clear the low bit of the function address.

     NOTE: got_addr is from GOT table whose lsb is always set by linker if it's
     Thumb function address.  PCREL_ADDR comes from PC-relative calculation
     which will finish during assembling.  GAS assembler before the fix for
     PR gas/21458 was not setting the lsb but does after that.  Always do the
     strip for both, so the code works with various combinations of glibc and
     Binutils.  */
  got_addr &= ~(Elf32_Addr) 1;
  pcrel_addr &= ~(Elf32_Addr) 1;
#endif
  return pcrel_addr - got_addr;
}


/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((unused))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  Elf32_Addr *got;
  extern void _dl_runtime_resolve (Elf32_Word);
  extern void _dl_runtime_profile (Elf32_Word);

  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* patb: this is different than i386 */
      /* The GOT entries for functions in the PLT have not yet been filled
	 in.  Their initial contents will arrange when called to push an
	 index into the .got section, load ip with &_GLOBAL_OFFSET_TABLE_[3],
	 and then jump to _GLOBAL_OFFSET_TABLE[2].  */
      got = (Elf32_Addr *) D_PTR (l, l_info[DT_PLTGOT]);
      /* If a library is prelinked but we have to relocate anyway,
	 we have to be able to undo the prelinking of .got.plt.
	 The prelinker saved us here address of .plt.  */
      if (got[1])
	l->l_mach.plt = got[1] + l->l_addr;
      got[1] = (Elf32_Addr) l;	/* Identify this shared object.  */

      /* The got[2] entry contains the address of a function which gets
	 called to get the address of a so far unresolved function and
	 jump to it.  The profiling extension of the dynamic linker allows
	 to intercept the calls to collect information.  In this case we
	 don't store the address in the GOT so that all future calls also
	 end in this function.  */
      if (profile)
	{
	  got[2] = (Elf32_Addr) &_dl_runtime_profile;

	  if (GLRO(dl_profile) != NULL
	      && _dl_name_match_p (GLRO(dl_profile), l))
	    /* Say that we really want profiling and the timers are
	       started.  */
	    GL(dl_profile_map) = l;
	}
      else
	/* This function will get called to fix up the GOT entry indicated by
	   the offset on the stack, and then jump to the resolved address.  */
	got[2] = (Elf32_Addr) &_dl_runtime_resolve;
    }

  return lazy;
}

#if defined(ARCH_HAS_BX)
#define BX(x) "bx\t" #x
#else
#define BX(x) "mov\tpc, " #x
#endif

/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK	0xf8000000UL

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_START asm ("\
.text\n\
.globl _start\n\
.type _start, %function\n\
.globl _dl_start_user\n\
.type _dl_start_user, %function\n\
_start:\n\
	@ we are PIC code, so get global offset table\n\
	ldr	sl, .L_GET_GOT\n\
	@ See if we were run as a command with the executable file\n\
	@ name as an extra leading argument.\n\
	ldr	r4, .L_SKIP_ARGS\n\
	@ at start time, all the args are on the stack\n\
	mov	r0, sp\n\
	bl	_dl_start\n\
	@ returns user entry point in r0\n\
_dl_start_user:\n\
	adr	r6, .L_GET_GOT\n\
	add	sl, sl, r6\n\
	ldr	r4, [sl, r4]\n\
	@ save the entry point in another register\n\
	mov	r6, r0\n\
	@ get the original arg count\n\
	ldr	r1, [sp]\n\
	@ get the argv address\n\
	add	r2, sp, #4\n\
	@ Fix up the stack if necessary.\n\
	cmp	r4, #0\n\
	bne	.L_fixup_stack\n\
.L_done_fixup:\n\
	@ compute envp\n\
	add	r3, r2, r1, lsl #2\n\
	add	r3, r3, #4\n\
	@ now we call _dl_init\n\
	ldr	r0, .L_LOADED\n\
	ldr	r0, [sl, r0]\n\
	@ call _dl_init\n\
	bl	_dl_init(PLT)\n\
	@ load the finalizer function\n\
	ldr	r0, .L_FINI_PROC\n\
	add	r0, sl, r0\n\
	@ jump to the user_s entry point\n\
	" BX(r6) "\n\
\n\
	@ iWMMXt and EABI targets require the stack to be eight byte\n\
	@ aligned - shuffle arguments etc.\n\
.L_fixup_stack:\n\
	@ subtract _dl_skip_args from original arg count\n\
	sub	r1, r1, r4\n\
	@ store the new argc in the new stack location\n\
	str	r1, [sp]\n\
	@ find the first unskipped argument\n\
	mov	r3, r2\n\
	add	r4, r2, r4, lsl #2\n\
	@ shuffle argv down\n\
1:	ldr	r5, [r4], #4\n\
	str	r5, [r3], #4\n\
	cmp	r5, #0\n\
	bne	1b\n\
	@ shuffle envp down\n\
1:	ldr	r5, [r4], #4\n\
	str	r5, [r3], #4\n\
	cmp	r5, #0\n\
	bne	1b\n\
	@ shuffle auxv down\n\
1:	ldmia	r4!, {r0, r5}\n\
	stmia	r3!, {r0, r5}\n\
	cmp	r0, #0\n\
	bne	1b\n\
	@ Update _dl_argv\n\
	ldr	r3, .L_ARGV\n\
	str	r2, [sl, r3]\n\
	b	.L_done_fixup\n\
\n\
.L_GET_GOT:\n\
	.word	_GLOBAL_OFFSET_TABLE_ - .L_GET_GOT\n\
.L_SKIP_ARGS:\n\
	.word	_dl_skip_args(GOTOFF)\n\
.L_FINI_PROC:\n\
	.word	_dl_fini(GOTOFF)\n\
.L_ARGV:\n\
	.word	_dl_argv(GOTOFF)\n\
.L_LOADED:\n\
	.word	_rtld_local(GOTOFF)\n\
.previous\n\
");

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry or
   TLS variable, so undefined references should not be allowed to
   define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.
   ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA iff TYPE describes relocation against
   protected data whose address may be external due to copy relocation.  */
#ifndef RTLD_BOOTSTRAP
# define elf_machine_type_class(type) \
  ((((type) == R_ARM_JUMP_SLOT || (type) == R_ARM_TLS_DTPMOD32		\
     || (type) == R_ARM_TLS_DTPOFF32 || (type) == R_ARM_TLS_TPOFF32	\
     || (type) == R_ARM_TLS_DESC)					\
    * ELF_RTYPE_CLASS_PLT)						\
   | (((type) == R_ARM_COPY) * ELF_RTYPE_CLASS_COPY)			\
   | (((type) == R_ARM_GLOB_DAT) * ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA))
#else
#define elf_machine_type_class(type) \
  ((((type) == R_ARM_JUMP_SLOT) * ELF_RTYPE_CLASS_PLT)	\
   | (((type) == R_ARM_COPY) * ELF_RTYPE_CLASS_COPY)	\
   | (((type) == R_ARM_GLOB_DAT) * ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA))
#endif

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	R_ARM_JUMP_SLOT

/* ARM never uses Elf32_Rela relocations for the dynamic linker.
   Prelinked libraries may use Elf32_Rela though.  */
#define ELF_MACHINE_PLT_REL 1

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

#endif /* !dl_machine_h */


/* ARM never uses Elf32_Rela relocations for the dynamic linker.
   Prelinked libraries may use Elf32_Rela though.  */
#define ELF_MACHINE_NO_RELA defined RTLD_BOOTSTRAP
#define ELF_MACHINE_NO_REL 0

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER arm_gnu_pltenter
#define ARCH_LA_PLTEXIT arm_gnu_pltexit

#ifdef RESOLVE_MAP
/* Handle a PC24 reloc, including the out-of-range case.  */
auto void
relocate_pc24 (struct link_map *map, Elf32_Addr value,
               Elf32_Addr *const reloc_addr, Elf32_Sword addend)
{
  Elf32_Addr new_value;

  /* Set NEW_VALUE based on V, and return true iff it overflows 24 bits.  */
  inline bool set_new_value (Elf32_Addr v)
  {
    new_value = v + addend - (Elf32_Addr) reloc_addr;
    Elf32_Addr topbits = new_value & 0xfe000000;
    return topbits != 0xfe000000 && topbits != 0x00000000;
  }

  if (set_new_value (value))
    {
      /* The PC-relative address doesn't fit in 24 bits!  */

      static void *fix_page;
      static size_t fix_offset;
      if (fix_page == NULL)
        {
          void *new_page = __mmap (NULL, GLRO(dl_pagesize),
                                   PROT_READ | PROT_WRITE | PROT_EXEC,
                                   MAP_PRIVATE | MAP_ANON, -1, 0);
          if (new_page == MAP_FAILED)
            _dl_signal_error (0, map->l_name, NULL,
                              "could not map page for fixup");
          fix_page = new_page;
          assert (fix_offset == 0);
        }

      Elf32_Word *fix_address = fix_page + fix_offset;
      fix_address[0] = 0xe51ff004;	/* ldr pc, [pc, #-4] */
      fix_address[1] = value;

      fix_offset += sizeof fix_address[0] * 2;
      if (fix_offset >= GLRO(dl_pagesize))
        {
          fix_page = NULL;
          fix_offset = 0;
        }

      if (set_new_value ((Elf32_Addr) fix_address))
        _dl_signal_error (0, map->l_name, NULL,
                          "R_ARM_PC24 relocation out of range");
    }

  *reloc_addr = (*reloc_addr & 0xff000000) | ((new_value >> 2) & 0x00ffffff);
}

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

auto inline void
__attribute__ ((always_inline))
elf_machine_rel (struct link_map *map, const Elf32_Rel *reloc,
		 const Elf32_Sym *sym, const struct r_found_version *version,
		 void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);

#if !defined RTLD_BOOTSTRAP || !defined HAVE_Z_COMBRELOC
  if (__builtin_expect (r_type == R_ARM_RELATIVE, 0))
    {
# if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC
      /* This is defined in rtld.c, but nowhere in the static libc.a;
	 make the reference weak so static programs can still link.
	 This declaration cannot be done when compiling rtld.c
	 (i.e. #ifdef RTLD_BOOTSTRAP) because rtld.c contains the
	 common defn for _dl_rtld_map, which is incompatible with a
	 weak decl in the same file.  */
#  ifndef SHARED
      weak_extern (_dl_rtld_map);
#  endif
      if (map != &GL(dl_rtld_map)) /* Already done in rtld itself.  */
# endif
	*reloc_addr += map->l_addr;
    }
# ifndef RTLD_BOOTSTRAP
  else if (__builtin_expect (r_type == R_ARM_NONE, 0))
    return;
# endif
  else
#endif
    {
      const Elf32_Sym *const refsym = sym;
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf32_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      if (sym != NULL
	  && __builtin_expect (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC, 0)
	  && __builtin_expect (sym->st_shndx != SHN_UNDEF, 1)
	  && __builtin_expect (!skip_ifunc, 1))
	value = elf_ifunc_invoke (value);

      switch (r_type)
	{
	case R_ARM_COPY:
	  if (sym == NULL)
	    /* This can happen in trace mode if an object could not be
	       found.  */
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
		  MIN (sym->st_size, refsym->st_size));
	  break;
	case R_ARM_GLOB_DAT:
	case R_ARM_JUMP_SLOT:
# ifdef RTLD_BOOTSTRAP
	  /* Fix weak undefined references.  */
	  if (sym != NULL && sym->st_value == 0)
	    *reloc_addr = 0;
	  else
# endif
	    *reloc_addr = value;
	  break;
	case R_ARM_ABS32:
	  {
	    struct unaligned
	      {
		Elf32_Addr x;
	      } __attribute__ ((packed, may_alias));
# ifndef RTLD_BOOTSTRAP
	   /* This is defined in rtld.c, but nowhere in the static
	      libc.a; make the reference weak so static programs can
	      still link.  This declaration cannot be done when
	      compiling rtld.c (i.e.  #ifdef RTLD_BOOTSTRAP) because
	      rtld.c contains the common defn for _dl_rtld_map, which
	      is incompatible with a weak decl in the same file.  */
#  ifndef SHARED
	    weak_extern (_dl_rtld_map);
#  endif
	    if (map == &GL(dl_rtld_map))
	      /* Undo the relocation done here during bootstrapping.
		 Now we will relocate it anew, possibly using a
		 binding found in the user program or a loaded library
		 rather than the dynamic linker's built-in definitions
		 used while loading those libraries.  */
	      value -= SYMBOL_ADDRESS (map, refsym, true);
# endif
	    /* Support relocations on mis-aligned offsets.  */
	    ((struct unaligned *) reloc_addr)->x += value;
	    break;
	  }
	case R_ARM_TLS_DESC:
	  {
	    struct tlsdesc *td = (struct tlsdesc *)reloc_addr;

# ifndef RTLD_BOOTSTRAP
	    if (! sym)
	      td->entry = _dl_tlsdesc_undefweak;
	    else
# endif
	      {
		if (ELF32_R_SYM (reloc->r_info) == STN_UNDEF)
		  value = td->argument.value;
		else
		  value = sym->st_value;

# ifndef RTLD_BOOTSTRAP
#  ifndef SHARED
		CHECK_STATIC_TLS (map, sym_map);
#  else
		if (!TRY_STATIC_TLS (map, sym_map))
		  {
		    td->argument.pointer
		      = _dl_make_tlsdesc_dynamic (sym_map, value);
		    td->entry = _dl_tlsdesc_dynamic;
		  }
		else
#  endif
# endif
		{
		  td->argument.value = value + sym_map->l_tls_offset;
		  td->entry = _dl_tlsdesc_return;
		}
	      }
	    }
	    break;
	case R_ARM_PC24:
          relocate_pc24 (map, value, reloc_addr,
                         /* Sign-extend the 24-bit addend in the
                            instruction (which counts instructions), and
                            then shift it up two so as to count bytes.  */
                         (((Elf32_Sword) *reloc_addr << 8) >> 8) << 2);
	  break;
#if !defined RTLD_BOOTSTRAP
	case R_ARM_TLS_DTPMOD32:
	  /* Get the information from the link map returned by the
	     resolv function.  */
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
	  break;

	case R_ARM_TLS_DTPOFF32:
	  if (sym != NULL)
	    *reloc_addr += sym->st_value;
	  break;

	case R_ARM_TLS_TPOFF32:
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr += sym->st_value + sym_map->l_tls_offset;
	    }
	  break;
	case R_ARM_IRELATIVE:
	  value = map->l_addr + *reloc_addr;
	  if (__glibc_likely (!skip_ifunc))
	    value = ((Elf32_Addr (*) (int)) value) (GLRO(dl_hwcap));
	  *reloc_addr = value;
	  break;
#endif
	default:
	  _dl_reloc_bad_type (map, r_type, 0);
	  break;
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

  if (__builtin_expect (r_type == R_ARM_RELATIVE, 0))
    *reloc_addr = map->l_addr + reloc->r_addend;
  else if (__builtin_expect (r_type == R_ARM_NONE, 0))
    return;
  else
    {
# ifndef RESOLVE_CONFLICT_FIND_MAP
      const Elf32_Sym *const refsym = sym;
# endif
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf32_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      if (sym != NULL
	  && __builtin_expect (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC, 0)
	  && __builtin_expect (sym->st_shndx != SHN_UNDEF, 1)
	  && __builtin_expect (!skip_ifunc, 1))
	value = elf_ifunc_invoke (value);

      switch (r_type)
	{
#  ifndef RESOLVE_CONFLICT_FIND_MAP
	  /* Not needed for dl-conflict.c.  */
	case R_ARM_COPY:
	  if (sym == NULL)
	    /* This can happen in trace mode if an object could not be
	       found.  */
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
		  MIN (sym->st_size, refsym->st_size));
	  break;
#  endif /* !RESOLVE_CONFLICT_FIND_MAP */
	case R_ARM_GLOB_DAT:
	case R_ARM_JUMP_SLOT:
	case R_ARM_ABS32:
	  *reloc_addr = value + reloc->r_addend;
	  break;
	case R_ARM_PC24:
          relocate_pc24 (map, value, reloc_addr, reloc->r_addend);
	  break;
#if !defined RTLD_BOOTSTRAP
	case R_ARM_TLS_DTPMOD32:
	  /* Get the information from the link map returned by the
	     resolv function.  */
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
	  break;

	case R_ARM_TLS_DTPOFF32:
	  *reloc_addr = (sym == NULL ? 0 : sym->st_value) + reloc->r_addend;
	  break;

	case R_ARM_TLS_TPOFF32:
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = (sym->st_value + sym_map->l_tls_offset
			     + reloc->r_addend);
	    }
	  break;
	case R_ARM_IRELATIVE:
	  value = map->l_addr + reloc->r_addend;
	  if (__glibc_likely (!skip_ifunc))
	    value = ((Elf32_Addr (*) (int)) value) (GLRO(dl_hwcap));
	  *reloc_addr = value;
	  break;
#endif
	default:
	  _dl_reloc_bad_type (map, r_type, 0);
	  break;
	}
    }
}
# endif

auto inline void
__attribute__ ((always_inline))
elf_machine_rel_relative (Elf32_Addr l_addr, const Elf32_Rel *reloc,
			  void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
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
# endif

auto inline void
__attribute__ ((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rel *reloc,
		      int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);
  /* Check for unexpected PLT reloc type.  */
  if (__builtin_expect (r_type == R_ARM_JUMP_SLOT, 1))
    {
      if (__builtin_expect (map->l_mach.plt, 0) == 0)
	*reloc_addr += l_addr;
      else
	*reloc_addr = map->l_mach.plt;
    }
  else if (__builtin_expect (r_type == R_ARM_TLS_DESC, 1))
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
      elf_machine_rel (map, reloc, sym, version, reloc_addr, skip_ifunc);
    }
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

#endif /* RESOLVE_MAP */
