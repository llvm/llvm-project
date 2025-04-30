/* Machine-dependent ELF dynamic relocation inline functions.  C-SKY version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define ELF_MACHINE_NAME "csky"

#include <sys/param.h>
#include <sysdep.h>
#include <dl-tls.h>

/* Return nonzero if ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_CSKY;
}

/* Return the link-time address of _DYNAMIC.
   This must be inlined in a function which uses global data.  */
static inline Elf32_Addr
elf_machine_dynamic (void)
{
  register Elf32_Addr *got __asm__ ("gb");
  return *got;
}

/* Return the run-time load address ,of the shared object.  */
static inline Elf32_Addr
elf_machine_load_address (void)
{
  extern Elf32_Addr __dl_start (void *) asm ("_dl_start");
  Elf32_Addr got_addr = (Elf32_Addr) &__dl_start;
  Elf32_Addr pcrel_addr;
  asm  ("grs %0,_dl_start\n" : "=r" (pcrel_addr));

  return pcrel_addr - got_addr;
}


/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((always_inline))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  Elf32_Addr *got;
  extern void _dl_runtime_resolve (Elf32_Word);

  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* The GOT entries for functions in the PLT have not yet been
	 filled in.  Their initial contents will arrange when called
	 to push an offset into the .rela.plt section, push
	 _GLOBAL_OFFSET_TABLE_[1], and then jump to
	 _GLOBAL_OFFSET_TABLE_[2].  */
      got = (Elf32_Addr *) D_PTR (l, l_info[DT_PLTGOT]);

      if (got[1])
	l->l_mach.plt = got[1] + l->l_addr;
      got[1] = (Elf32_Addr) l; /* Identify this shared object.  */

      /* The got[2] entry contains the address of a function which gets
	 called to get the address of a so far unresolved function and
	 jump to it.  The profiling extension of the dynamic linker allows
	 to intercept the calls to collect information.  In this case we
	 don't store the address in the GOT so that all future calls also
	 end in this function.  */
	got[2] = (Elf32_Addr) &_dl_runtime_resolve;
    }
  return lazy;
}

/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK 0x80000000UL

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */
#define RTLD_START asm ("\
.text\n\
.globl _start\n\
.type _start, @function\n\
.globl _dl_start_user\n\
.type _dl_start_user, @function\n\
_start:\n\
	grs	gb, .Lgetpc1\n\
.Lgetpc1:\n\
	lrw	t0, .Lgetpc1@GOTPC\n\
	addu	gb, t0\n\
	mov	a0, sp\n\
	lrw	t1, _dl_start@GOTOFF\n\
	addu	t1, gb\n\
	jsr	t1\n\
_dl_start_user:\n\
	/* get _dl_skip_args */    \n\
	lrw	r11, _dl_skip_args@GOTOFF\n\
	addu	r11, gb\n\
	ldw	r11, (r11, 0)\n\
	/* store program entry address in r11 */ \n\
	mov	r10, a0\n\
	/* Get argc */\n\
	ldw	a1, (sp, 0)\n\
	/* Get **argv */\n\
	mov	a2, sp\n\
	addi	a2, 4\n\
	cmpnei	r11, 0\n\
	bt	.L_fixup_stack\n\
.L_done_fixup:\n\
	mov	a3, a1\n\
	lsli	a3, 2\n\
	add	a3, a2\n\
	addi	a3, 4\n\
	lrw	a0, _rtld_local@GOTOFF\n\
	addu	a0, gb\n\
	ldw	a0, (a0, 0)\n\
	lrw	t1, _dl_init@PLT\n\
	addu	t1, gb\n\
	ldw	t1, (t1)\n\
	jsr	t1\n\
	lrw	a0, _dl_fini@GOTOFF\n\
	addu	a0, gb\n\
	jmp	r10\n\
.L_fixup_stack:\n\
	subu	a1, r11\n\
	lsli	r11, 2\n\
	addu	sp, r11\n\
	stw	a1, (sp, 0)\n\
	mov	a2, sp\n\
	addi	a2, 4\n\
	lrw	a3, _dl_argv@GOTOFF\n\
	addu	a3, gb\n\
	stw	a2, (a3, 0)\n\
	br	.L_done_fixup\n\
");

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry or
   TLS variable, so undefined references should not be allowed to
   define the value.
   ELF_RTYPE_CLASS_NOCOPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#ifndef RTLD_BOOTSTRAP
# define elf_machine_type_class(type) \
  ((((type) == R_CKCORE_JUMP_SLOT || (type) == R_CKCORE_TLS_DTPMOD32	   \
     || (type) == R_CKCORE_TLS_DTPOFF32 || (type) == R_CKCORE_TLS_TPOFF32) \
    * ELF_RTYPE_CLASS_PLT)						   \
   | (((type) == R_CKCORE_COPY) * ELF_RTYPE_CLASS_COPY))
#else
# define elf_machine_type_class(type) \
  ((((type) == R_CKCORE_JUMP_SLOT     \
   | (((type) == R_CKCORE_COPY) * ELF_RTYPE_CLASS_COPY))
#endif

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT R_CKCORE_JUMP_SLOT

/* C-SKY never uses Elf32_Rel relocations.  */
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

/* Return the final value of a plt relocation.  On the csky the JMP_SLOT
   relocation ignores the addend.  */
static inline Elf32_Addr
elf_machine_plt_value (struct link_map *map, const Elf32_Rela *reloc,
		       Elf32_Addr value)
{
  return value;
}

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER csky_gnu_pltenter
#define ARCH_LA_PLTEXIT csky_gnu_pltexit

#endif /* !dl_machine_h */
#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

auto inline void __attribute__ ((unused, always_inline))
elf_machine_rela (struct link_map *map, const Elf32_Rela *reloc,
		  const Elf32_Sym *sym, const struct r_found_version *version,
		  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);
  unsigned short __attribute__ ((unused)) *opcode16_addr;
  Elf32_Addr __attribute__ ((unused)) insn_opcode = 0x0;

  if (__builtin_expect (r_type == R_CKCORE_RELATIVE, 0))
    *reloc_addr = map->l_addr + reloc->r_addend;
  else
    {
      const Elf32_Sym *const refsym = sym;
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      ElfW(Addr) value = SYMBOL_ADDRESS (sym_map, sym, true);
      opcode16_addr = (unsigned short *)reloc_addr;

      switch (r_type)
	{
	case R_CKCORE_COPY:
	  if (sym == NULL)
	    /* This can happen in trace mode if an object could not be
	       found.  */
	    break;
	  if (sym->st_size > refsym->st_size
	      || (sym->st_size < refsym->st_size && GLRO(dl_verbose)))
	    {
	      const char *strtab;

	      strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
	      _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				rtld_progname ?: "<program name unknown>",
				strtab + refsym->st_name);
	    }
	  memcpy (reloc_addr_arg, (void *) value,
		  MIN (sym->st_size, refsym->st_size));
	  break;
	case R_CKCORE_GLOB_DAT:
	case R_CKCORE_JUMP_SLOT:
	  *reloc_addr = value;
	  break;
	case R_CKCORE_ADDR32:
	  *reloc_addr = value + reloc->r_addend;
	  break;
	case R_CKCORE_PCREL32:
	  *reloc_addr = value + reloc->r_addend - (Elf32_Addr) reloc_addr;
	  break;
#if defined(__CK810__) || defined(__CK807__)
	case R_CKCORE_ADDR_HI16:
	  insn_opcode = (*opcode16_addr << 16) | (*(opcode16_addr + 1));
	  insn_opcode = (insn_opcode & 0xffff0000)
			    | (((value + reloc->r_addend) >> 16) & 0xffff);
	  *(opcode16_addr++) = (unsigned short)(insn_opcode >> 16);
	  *opcode16_addr = (unsigned short)(insn_opcode & 0xffff);
	  break;
	case R_CKCORE_ADDR_LO16:
	  insn_opcode = (*opcode16_addr << 16) | (*(opcode16_addr + 1));
	  insn_opcode = (insn_opcode & 0xffff0000)
			    | ((value + reloc->r_addend) & 0xffff);
	   *(opcode16_addr++) = (unsigned short)(insn_opcode >> 16);
	   *opcode16_addr = (unsigned short)(insn_opcode & 0xffff);
	   break;
	case R_CKCORE_PCREL_IMM26BY2:
	{
	  unsigned int offset = ((value + reloc->r_addend
				  - (unsigned int)reloc_addr) >> 1);
	  insn_opcode = (*opcode16_addr << 16) | (*(opcode16_addr + 1));
	  if (offset > 0x3ffffff){
	    const char *strtab;
	    strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);

	    _dl_error_printf ("\
%s:The reloc R_CKCORE_PCREL_IMM26BY2 cannot reach the symbol '%s'.\n",
	      rtld_progname ?: "<program name unknown>",
	      strtab + refsym->st_name);
	    break;
	  }
	  insn_opcode = (insn_opcode & ~0x3ffffff) | offset;
	  *(opcode16_addr++) = (unsigned short)(insn_opcode >> 16);
	  *opcode16_addr = (unsigned short)(insn_opcode & 0xffff);
	  break;
	}
	case R_CKCORE_PCREL_JSR_IMM26BY2:
	  break;
#endif
#ifndef RTLD_BOOTSTRAP
	case R_CKCORE_TLS_DTPMOD32:
	/* Get the information from the link map returned by the
	   resolv function.  */
	  if (sym_map != NULL)
	    *reloc_addr = sym_map->l_tls_modid;
	  break;
	case R_CKCORE_TLS_DTPOFF32:
	  if (sym != NULL)
	    *reloc_addr =(sym == NULL ? 0 : sym->st_value) + reloc->r_addend;
	  break;
	case R_CKCORE_TLS_TPOFF32:
	  if (sym != NULL)
	    {
	      CHECK_STATIC_TLS (map, sym_map);
	      *reloc_addr = (sym->st_value + sym_map->l_tls_offset
			     + reloc->r_addend);
	    }
	  break;
#endif /* !RTLD_BOOTSTRAP */
	case R_CKCORE_NONE:
	  break;
	default:
	  break;
	}
    }
}

auto inline void __attribute__ ((unused, always_inline))
elf_machine_rela_relative (Elf32_Addr l_addr, const Elf32_Rela *reloc,
			   void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  *reloc_addr = l_addr + reloc->r_addend;
}

auto inline void __attribute__ ((unused, always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rela *reloc,
		      int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);
  if (ELF32_R_TYPE (reloc->r_info) == R_CKCORE_JUMP_SLOT)
    {
      /* Check for unexpected PLT reloc type.  */
      if (__builtin_expect (r_type == R_CKCORE_JUMP_SLOT, 1))
	{
	  if (__builtin_expect (map->l_mach.plt, 0) == 0)
	    *reloc_addr = l_addr + reloc->r_addend;
	  else
	    *reloc_addr = map->l_mach.plt;
	}
    }
}

#endif /* RESOLVE_MAP */
