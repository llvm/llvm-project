/* Machine-dependent ELF dynamic relocation inline functions.  Sparc64 version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#define ELF_MACHINE_NAME "sparc64"

#include <string.h>
#include <sys/param.h>
#include <ldsodefs.h>
#include <sysdep.h>
#include <dl-plt.h>

#define ELF64_R_TYPE_ID(info)	((info) & 0xff)
#define ELF64_R_TYPE_DATA(info) ((info) >> 8)

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf64_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_SPARCV9;
}

/* We have to do this because elf_machine_{dynamic,load_address} can be
   invoked from functions that have no GOT references, and thus the compiler
   has no obligation to load the PIC register.  */
#define LOAD_PIC_REG(PIC_REG)	\
do {	Elf64_Addr tmp;		\
	__asm("sethi %%hi(_GLOBAL_OFFSET_TABLE_-4), %1\n\t" \
	      "rd %%pc, %0\n\t" \
	      "add %1, %%lo(_GLOBAL_OFFSET_TABLE_+4), %1\n\t" \
	      "add %0, %1, %0" \
	      : "=r" (PIC_REG), "=r" (tmp)); \
} while (0)

/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  This must be inlined in a function which
   uses global data.  */
static inline Elf64_Addr
elf_machine_dynamic (void)
{
  register Elf64_Addr *elf_pic_register __asm__("%l7");

  LOAD_PIC_REG (elf_pic_register);

  return *elf_pic_register;
}

/* Return the run-time load address of the shared object.  */
static inline Elf64_Addr
elf_machine_load_address (void)
{
  register Elf32_Addr *pc __asm ("%o7");
  register Elf64_Addr *got __asm ("%l7");

  __asm ("sethi %%hi(_GLOBAL_OFFSET_TABLE_-4), %1\n\t"
	 "call 1f\n\t"
	 " add %1, %%lo(_GLOBAL_OFFSET_TABLE_+4), %1\n\t"
	 "call _DYNAMIC\n\t"
	 "call _GLOBAL_OFFSET_TABLE_\n"
	 "1:\tadd %1, %0, %1\n\t" : "=r" (pc), "=r" (got));

  /* got is now l_addr + _GLOBAL_OFFSET_TABLE_
     *got is _DYNAMIC
     pc[2]*4 is l_addr + _DYNAMIC - (long)pc - 8
     pc[3]*4 is l_addr + _GLOBAL_OFFSET_TABLE_ - (long)pc - 12  */
  return (Elf64_Addr) got - *got + (Elf32_Sword) ((pc[2] - pc[3]) * 4) - 4;
}

static inline Elf64_Addr __attribute__ ((always_inline))
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf64_Rela *reloc,
		       Elf64_Addr *reloc_addr, Elf64_Addr value)
{
  sparc64_fixup_plt (map, reloc, reloc_addr, value + reloc->r_addend,
		     reloc->r_addend, 1);
  return value;
}

/* Return the final value of a plt relocation.  */
static inline Elf64_Addr
elf_machine_plt_value (struct link_map *map, const Elf64_Rela *reloc,
		       Elf64_Addr value)
{
  /* Don't add addend here, but in elf_machine_fixup_plt instead.
     value + reloc->r_addend is the value which should actually be
     stored into .plt data slot.  */
  return value;
}

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry, so
   PLT entries should not be allowed to define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#define elf_machine_type_class(type) \
  ((((type) == R_SPARC_JMP_SLOT						      \
     || ((type) >= R_SPARC_TLS_GD_HI22 && (type) <= R_SPARC_TLS_TPOFF64))     \
    * ELF_RTYPE_CLASS_PLT)						      \
   | (((type) == R_SPARC_COPY) * ELF_RTYPE_CLASS_COPY))

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT	R_SPARC_JMP_SLOT

/* The SPARC never uses Elf64_Rel relocations.  */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  if (l->l_info[DT_JMPREL] && lazy)
    {
      extern void _dl_runtime_resolve_0 (void);
      extern void _dl_runtime_resolve_1 (void);
      extern void _dl_runtime_profile_0 (void);
      extern void _dl_runtime_profile_1 (void);
      Elf64_Addr res0_addr, res1_addr;
      unsigned int *plt = (void *) D_PTR (l, l_info[DT_PLTGOT]);

      if (__builtin_expect(profile, 0))
	{
	  res0_addr = (Elf64_Addr) &_dl_runtime_profile_0;
	  res1_addr = (Elf64_Addr) &_dl_runtime_profile_1;

	  if (GLRO(dl_profile) != NULL
	      && _dl_name_match_p (GLRO(dl_profile), l))
	    GL(dl_profile_map) = l;
	}
      else
	{
	  res0_addr = (Elf64_Addr) &_dl_runtime_resolve_0;
	  res1_addr = (Elf64_Addr) &_dl_runtime_resolve_1;
	}

      /* PLT0 looks like:

	 sethi	%uhi(_dl_runtime_{resolve,profile}_0), %g4
	 sethi	%hi(_dl_runtime_{resolve,profile}_0), %g5
	 or	%g4, %ulo(_dl_runtime_{resolve,profile}_0), %g4
	 or	%g5, %lo(_dl_runtime_{resolve,profile}_0), %g5
	 sllx	%g4, 32, %g4
	 add	%g4, %g5, %g5
	 jmpl	%g5, %g4
	  nop
       */

      plt[0] = 0x09000000 | (res0_addr >> (64 - 22));
      plt[1] = 0x0b000000 | ((res0_addr >> 10) & 0x003fffff);
      plt[2] = 0x88112000 | ((res0_addr >> 32) & 0x3ff);
      plt[3] = 0x8a116000 | (res0_addr & 0x3ff);
      plt[4] = 0x89293020;
      plt[5] = 0x8a010005;
      plt[6] = 0x89c14000;
      plt[7] = 0x01000000;

      /* PLT1 looks like:

	 sethi	%uhi(_dl_runtime_{resolve,profile}_1), %g4
	 sethi	%hi(_dl_runtime_{resolve,profile}_1), %g5
	 or	%g4, %ulo(_dl_runtime_{resolve,profile}_1), %g4
	 or	%g5, %lo(_dl_runtime_{resolve,profile}_1), %g5
	 sllx	%g4, 32, %g4
	 add	%g4, %g5, %g5
	 jmpl	%g5, %g4
	  nop
       */

      plt[8] = 0x09000000 | (res1_addr >> (64 - 22));
      plt[9] = 0x0b000000 | ((res1_addr >> 10) & 0x003fffff);
      plt[10] = 0x88112000 | ((res1_addr >> 32) & 0x3ff);
      plt[11] = 0x8a116000 | (res1_addr & 0x3ff);
      plt[12] = 0x89293020;
      plt[13] = 0x8a010005;
      plt[14] = 0x89c14000;
      plt[15] = 0x01000000;

      /* Now put the magic cookie at the beginning of .PLT2
	 Entry .PLT3 is unused by this implementation.  */
      *((struct link_map **)(&plt[16])) = l;

      if (__builtin_expect (l->l_info[VALIDX(DT_GNU_PRELINKED)] != NULL, 0)
	  || __builtin_expect (l->l_info [VALIDX (DT_GNU_LIBLISTSZ)] != NULL, 0))
	{
	  /* Need to reinitialize .plt to undo prelinking.  */
	  Elf64_Rela *rela = (Elf64_Rela *) D_PTR (l, l_info[DT_JMPREL]);
	  Elf64_Rela *relaend
	    = (Elf64_Rela *) ((char *) rela
			      + l->l_info[DT_PLTRELSZ]->d_un.d_val);

	  /* prelink must ensure there are no R_SPARC_NONE relocs left
	     in .rela.plt.  */
	  while (rela < relaend)
	    {
	      if (__builtin_expect (rela->r_addend, 0) != 0)
		{
		  Elf64_Addr slot = ((rela->r_offset + l->l_addr + 0x400
				      - (Elf64_Addr) plt)
				     / 0x1400) * 0x1400
				    + (Elf64_Addr) plt - 0x400;
		  /* ldx [%o7 + X], %g1  */
		  unsigned int first_ldx = *(unsigned int *)(slot + 12);
		  Elf64_Addr ptr = slot + (first_ldx & 0xfff) + 4;

		  *(Elf64_Addr *) (rela->r_offset + l->l_addr)
		    = (Elf64_Addr) plt
		      - (slot + ((rela->r_offset + l->l_addr - ptr) / 8) * 24
			 + 4);
		  ++rela;
		  continue;
		}

	      *(unsigned int *) (rela->r_offset + l->l_addr)
		= 0x03000000 | (rela->r_offset + l->l_addr - (Elf64_Addr) plt);
	      *(unsigned int *) (rela->r_offset + l->l_addr + 4)
		= 0x30680000 | ((((Elf64_Addr) plt + 32 - rela->r_offset
				  - l->l_addr - 4) >> 2) & 0x7ffff);
	      __asm __volatile ("flush %0" : : "r" (rela->r_offset
						    + l->l_addr));
	      __asm __volatile ("flush %0+4" : : "r" (rela->r_offset
						      + l->l_addr));
	      ++rela;
	    }
	}
    }

  return lazy;
}

/* The PLT uses Elf64_Rela relocs.  */
#define elf_machine_relplt elf_machine_rela

/* Undo the sub %sp, 6*8, %sp; add %sp, STACK_BIAS + 22*8, %o0 below
   (but w/o STACK_BIAS) to get at the value we want in __libc_stack_end.  */
#define DL_STACK_END(cookie) \
  ((void *) (((long) (cookie)) - (22 - 6) * 8))

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_GOT_ADDRESS(pic_reg, reg, symbol)	\
	"sethi	%gdop_hix22(" #symbol "), " #reg "\n\t" \
	"xor	" #reg ", %gdop_lox10(" #symbol "), " #reg "\n\t" \
	"ldx	[" #pic_reg " + " #reg "], " #reg ", %gdop(" #symbol ")\n"

#define __S1(x)	#x
#define __S(x)	__S1(x)

#define RTLD_START __asm__ ( "\n"					\
"	.text\n"							\
"	.global	_start\n"						\
"	.type	_start, @function\n"					\
"	.align	32\n"							\
"_start:\n"								\
"   /* Make room for functions to drop their arguments on the stack.  */\n" \
"	sub	%sp, 6*8, %sp\n"					\
"   /* Pass pointer to argument block to _dl_start.  */\n"		\
"	call	_dl_start\n"						\
"	 add	 %sp," __S(STACK_BIAS) "+22*8,%o0\n"			\
"	/* FALLTHRU */\n"						\
"	.size _start, .-_start\n"					\
"\n"									\
"	.global	_dl_start_user\n"					\
"	.type	_dl_start_user, @function\n"				\
"_dl_start_user:\n"							\
"   /* Load the GOT register.  */\n"					\
"1:	call	11f\n"							\
"	 sethi	%hi(_GLOBAL_OFFSET_TABLE_-(1b-.)), %l7\n"		\
"11:	or	%l7, %lo(_GLOBAL_OFFSET_TABLE_-(1b-.)), %l7\n"		\
"	add	%l7, %o7, %l7\n"					\
"   /* Save the user entry point address in %l0.  */\n"			\
"	mov	%o0, %l0\n"						\
"   /* See if we were run as a command with the executable file name as an\n" \
"      extra leading argument.  If so, we must shift things around since we\n" \
"      must keep the stack doubleword aligned.  */\n"			\
	RTLD_GOT_ADDRESS(%l7, %g5, _dl_skip_args)			\
"	ld	[%g5], %i0\n"						\
"	brz,pt	%i0, 2f\n"						\
"	 ldx	[%sp + " __S(STACK_BIAS) " + 22*8], %i5\n"		\
"	/* Find out how far to shift.  */\n"				\
"	sub	%i5, %i0, %i5\n"					\
"	sllx	%i0, 3, %l6\n"						\
	RTLD_GOT_ADDRESS(%l7, %l4, _dl_argv)				\
"	stx	%i5, [%sp + " __S(STACK_BIAS) " + 22*8]\n"		\
"	add	%sp, " __S(STACK_BIAS) " + 23*8, %i1\n"			\
"	add	%i1, %l6, %i2\n"					\
"	ldx	[%l4], %l5\n"						\
"	/* Copy down argv.  */\n"					\
"12:	ldx	[%i2], %i3\n"						\
"	add	%i2, 8, %i2\n"						\
"	stx	%i3, [%i1]\n"						\
"	brnz,pt	%i3, 12b\n"						\
"	 add	%i1, 8, %i1\n"						\
"	sub	%l5, %l6, %l5\n"					\
"	/* Copy down envp.  */\n"					\
"13:	ldx	[%i2], %i3\n"						\
"	add	%i2, 8, %i2\n"						\
"	stx	%i3, [%i1]\n"						\
"	brnz,pt	%i3, 13b\n"						\
"	 add	%i1, 8, %i1\n"						\
"	/* Copy down auxiliary table.  */\n"				\
"14:	ldx	[%i2], %i3\n"						\
"	ldx	[%i2 + 8], %i4\n"					\
"	add	%i2, 16, %i2\n"						\
"	stx	%i3, [%i1]\n"						\
"	stx	%i4, [%i1 + 8]\n"					\
"	brnz,pt	%i3, 14b\n"						\
"	 add	%i1, 16, %i1\n"						\
"	stx	%l5, [%l4]\n"						\
"  /* %o0 = _dl_loaded, %o1 = argc, %o2 = argv, %o3 = envp.  */\n"	\
"2:\t"	RTLD_GOT_ADDRESS(%l7, %o0, _rtld_local)				\
"	sllx	%i5, 3, %o3\n"						\
"	add	%sp, " __S(STACK_BIAS) " + 23*8, %o2\n"			\
"	add	%o3, 8, %o3\n"						\
"	mov	%i5, %o1\n"						\
"	add	%o2, %o3, %o3\n"					\
"	call	_dl_init\n"						\
"	 ldx	[%o0], %o0\n"						\
"   /* Pass our finalizer function to the user in %g1.  */\n"		\
       RTLD_GOT_ADDRESS(%l7, %g1, _dl_fini)				\
"  /* Jump to the user's entry point and deallocate the extra stack we got.  */\n" \
"	jmp	%l0\n"							\
"	 add	%sp, 6*8, %sp\n"					\
"	.size	_dl_start_user, . - _dl_start_user\n"			\
"	.previous\n");

#endif /* dl_machine_h */

#define ARCH_LA_PLTENTER	sparc64_gnu_pltenter
#define ARCH_LA_PLTEXIT		sparc64_gnu_pltexit

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
#if !defined RTLD_BOOTSTRAP && !defined RESOLVE_CONFLICT_FIND_MAP
  const Elf64_Sym *const refsym = sym;
#endif
  Elf64_Addr value;
  const unsigned long int r_type = ELF64_R_TYPE_ID (reloc->r_info);
#if !defined RESOLVE_CONFLICT_FIND_MAP
  struct link_map *sym_map = NULL;
#endif

#if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC
  /* This is defined in rtld.c, but nowhere in the static libc.a; make the
     reference weak so static programs can still link.  This declaration
     cannot be done when compiling rtld.c (i.e.  #ifdef RTLD_BOOTSTRAP)
     because rtld.c contains the common defn for _dl_rtld_map, which is
     incompatible with a weak decl in the same file.  */
  weak_extern (_dl_rtld_map);
#endif

  if (__glibc_unlikely (r_type == R_SPARC_NONE))
    return;

  if (__glibc_unlikely (r_type == R_SPARC_SIZE64))
    {
      *reloc_addr = sym->st_size + reloc->r_addend;
      return;
    }

#if !defined RTLD_BOOTSTRAP || !defined HAVE_Z_COMBRELOC
  if (__glibc_unlikely (r_type == R_SPARC_RELATIVE))
    {
# if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC
      if (map != &_dl_rtld_map) /* Already done in rtld itself. */
# endif
	*reloc_addr += map->l_addr + reloc->r_addend;
      return;
    }
#endif

#ifndef RESOLVE_CONFLICT_FIND_MAP
  if (__builtin_expect (ELF64_ST_BIND (sym->st_info) == STB_LOCAL, 0)
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
#else
  value = 0;
#endif

  value += reloc->r_addend;	/* Assume copy relocs have zero addend.  */

  if (sym != NULL
      && __builtin_expect (ELFW(ST_TYPE) (sym->st_info) == STT_GNU_IFUNC, 0)
      && __builtin_expect (sym->st_shndx != SHN_UNDEF, 1)
      && __builtin_expect (!skip_ifunc, 1))
    value = ((Elf64_Addr (*) (int)) value) (GLRO(dl_hwcap));

  switch (r_type)
    {
#if !defined RTLD_BOOTSTRAP && !defined RESOLVE_CONFLICT_FIND_MAP
    case R_SPARC_COPY:
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
#endif
    case R_SPARC_64:
    case R_SPARC_GLOB_DAT:
      *reloc_addr = value;
      break;
    case R_SPARC_IRELATIVE:
      if (__glibc_likely (!skip_ifunc))
	value = ((Elf64_Addr (*) (int)) value) (GLRO(dl_hwcap));
      *reloc_addr = value;
      break;
    case R_SPARC_JMP_IREL:
      if (__glibc_likely (!skip_ifunc))
	value = ((Elf64_Addr (*) (int)) value) (GLRO(dl_hwcap));
      /* 'high' is always zero, for large PLT entries the linker
	 emits an R_SPARC_IRELATIVE.  */
#ifdef RESOLVE_CONFLICT_FIND_MAP
      sparc64_fixup_plt (NULL, reloc, reloc_addr, value, 0, 0);
#else
      sparc64_fixup_plt (map, reloc, reloc_addr, value, 0, 0);
#endif
      break;
    case R_SPARC_JMP_SLOT:
#ifdef RESOLVE_CONFLICT_FIND_MAP
      /* R_SPARC_JMP_SLOT conflicts against .plt[32768+]
	 relocs should be turned into R_SPARC_64 relocs
	 in .gnu.conflict section.
	 r_addend non-zero does not mean it is a .plt[32768+]
	 reloc, instead it is the actual address of the function
	 to call.  */
      sparc64_fixup_plt (NULL, reloc, reloc_addr, value, 0, 0);
#else
      sparc64_fixup_plt (map, reloc, reloc_addr, value, reloc->r_addend, 0);
#endif
      break;
#ifndef RESOLVE_CONFLICT_FIND_MAP
    case R_SPARC_TLS_DTPMOD64:
      /* Get the information from the link map returned by the
	 resolv function.  */
      if (sym_map != NULL)
	*reloc_addr = sym_map->l_tls_modid;
      break;
    case R_SPARC_TLS_DTPOFF64:
      /* During relocation all TLS symbols are defined and used.
	 Therefore the offset is already correct.  */
      *reloc_addr = (sym == NULL ? 0 : sym->st_value) + reloc->r_addend;
      break;
    case R_SPARC_TLS_TPOFF64:
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
# ifndef RTLD_BOOTSTRAP
    case R_SPARC_TLS_LE_HIX22:
    case R_SPARC_TLS_LE_LOX10:
      if (sym != NULL)
	{
	  CHECK_STATIC_TLS (map, sym_map);
	  value = sym->st_value - sym_map->l_tls_offset
	    + reloc->r_addend;
	  if (r_type == R_SPARC_TLS_LE_HIX22)
	    *(unsigned int *)reloc_addr =
	      ((*(unsigned int *)reloc_addr & 0xffc00000)
	       | (((~value) >> 10) & 0x3fffff));
	  else
	    *(unsigned int *)reloc_addr =
	      ((*(unsigned int *)reloc_addr & 0xffffe000) | (value & 0x3ff)
	       | 0x1c00);
	}
      break;
# endif
#endif
#ifndef RTLD_BOOTSTRAP
    case R_SPARC_8:
      *(char *) reloc_addr = value;
      break;
    case R_SPARC_16:
      *(short *) reloc_addr = value;
      break;
    case R_SPARC_32:
      *(unsigned int *) reloc_addr = value;
      break;
    case R_SPARC_DISP8:
      *(char *) reloc_addr = (value - (Elf64_Addr) reloc_addr);
      break;
    case R_SPARC_DISP16:
      *(short *) reloc_addr = (value - (Elf64_Addr) reloc_addr);
      break;
    case R_SPARC_DISP32:
      *(unsigned int *) reloc_addr = (value - (Elf64_Addr) reloc_addr);
      break;
    case R_SPARC_DISP64:
      *reloc_addr = (value - (Elf64_Addr) reloc_addr);
      break;
    case R_SPARC_REGISTER:
      *reloc_addr = value;
      break;
    case R_SPARC_WDISP30:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & 0xc0000000)
	 | (((value - (Elf64_Addr) reloc_addr) >> 2) & 0x3fffffff));
      break;

      /* MEDLOW code model relocs */
    case R_SPARC_LO10:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & ~0x3ff)
	 | (value & 0x3ff));
      break;
    case R_SPARC_HI22:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & 0xffc00000)
	 | ((value >> 10) & 0x3fffff));
      break;
    case R_SPARC_OLO10:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & ~0x1fff)
	 | (((value & 0x3ff) + ELF64_R_TYPE_DATA (reloc->r_info)) & 0x1fff));
      break;

      /* ABS34 code model reloc */
    case R_SPARC_H34:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & 0xffc00000)
	 | ((value >> 12) & 0x3fffff));
      break;

      /* MEDMID code model relocs */
    case R_SPARC_H44:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & 0xffc00000)
	 | ((value >> 22) & 0x3fffff));
      break;
    case R_SPARC_M44:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & ~0x3ff)
	 | ((value >> 12) & 0x3ff));
      break;
    case R_SPARC_L44:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & ~0xfff)
	 | (value & 0xfff));
      break;

      /* MEDANY code model relocs */
    case R_SPARC_HH22:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & 0xffc00000)
	 | (value >> 42));
      break;
    case R_SPARC_HM10:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & ~0x3ff)
	 | ((value >> 32) & 0x3ff));
      break;
    case R_SPARC_LM22:
      *(unsigned int *) reloc_addr =
	((*(unsigned int *)reloc_addr & 0xffc00000)
	 | ((value >> 10) & 0x003fffff));
      break;
    case R_SPARC_UA16:
      ((unsigned char *) reloc_addr_arg) [0] = value >> 8;
      ((unsigned char *) reloc_addr_arg) [1] = value;
      break;
    case R_SPARC_UA32:
      ((unsigned char *) reloc_addr_arg) [0] = value >> 24;
      ((unsigned char *) reloc_addr_arg) [1] = value >> 16;
      ((unsigned char *) reloc_addr_arg) [2] = value >> 8;
      ((unsigned char *) reloc_addr_arg) [3] = value;
      break;
    case R_SPARC_UA64:
      if (! ((long) reloc_addr_arg & 3))
	{
	  /* Common in .eh_frame */
	  ((unsigned int *) reloc_addr_arg) [0] = value >> 32;
	  ((unsigned int *) reloc_addr_arg) [1] = value;
	  break;
	}
      ((unsigned char *) reloc_addr_arg) [0] = value >> 56;
      ((unsigned char *) reloc_addr_arg) [1] = value >> 48;
      ((unsigned char *) reloc_addr_arg) [2] = value >> 40;
      ((unsigned char *) reloc_addr_arg) [3] = value >> 32;
      ((unsigned char *) reloc_addr_arg) [4] = value >> 24;
      ((unsigned char *) reloc_addr_arg) [5] = value >> 16;
      ((unsigned char *) reloc_addr_arg) [6] = value >> 8;
      ((unsigned char *) reloc_addr_arg) [7] = value;
      break;
#endif
#if !defined RTLD_BOOTSTRAP || defined _NDEBUG
    default:
      _dl_reloc_bad_type (map, r_type, 0);
      break;
#endif
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

  if (__glibc_likely (r_type == R_SPARC_JMP_SLOT))
    ;
  else if (r_type == R_SPARC_JMP_IREL
	   || r_type == R_SPARC_IRELATIVE)
    {
      Elf64_Addr value = map->l_addr + reloc->r_addend;
      if (__glibc_likely (!skip_ifunc))
	value = ((Elf64_Addr (*) (int)) value) (GLRO(dl_hwcap));
      if (r_type == R_SPARC_JMP_IREL)
	{
	  /* 'high' is always zero, for large PLT entries the linker
	     emits an R_SPARC_IRELATIVE.  */
	  sparc64_fixup_plt (map, reloc, reloc_addr, value, 0, 1);
	}
      else
	*reloc_addr = value;
    }
  else if (r_type == R_SPARC_NONE)
    ;
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

#endif	/* RESOLVE_MAP */
