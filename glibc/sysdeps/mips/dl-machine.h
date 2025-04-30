/* Machine-dependent ELF dynamic relocation inline functions.  MIPS version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kazumoto Kojima <kkojima@info.kanagawa-u.ac.jp>.

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

/*  FIXME: Profiling of shared libraries is not implemented yet.  */
#ifndef dl_machine_h
#define dl_machine_h

#define ELF_MACHINE_NAME "MIPS"

#include <entry.h>

#ifndef ENTRY_POINT
#error ENTRY_POINT needs to be defined for MIPS.
#endif

#include <sgidefs.h>
#include <sysdep.h>
#include <sys/asm.h>
#include <dl-tls.h>

/* The offset of gp from GOT might be system-dependent.  It's set by
   ld.  The same value is also */
#define OFFSET_GP_GOT 0x7ff0

#ifndef _RTLD_PROLOGUE
# define _RTLD_PROLOGUE(entry)						\
	".globl\t" __STRING(entry) "\n\t"				\
	".ent\t" __STRING(entry) "\n\t"					\
	".type\t" __STRING(entry) ", @function\n"			\
	__STRING(entry) ":\n\t"
#endif

#ifndef _RTLD_EPILOGUE
# define _RTLD_EPILOGUE(entry)						\
	".end\t" __STRING(entry) "\n\t"					\
	".size\t" __STRING(entry) ", . - " __STRING(entry) "\n\t"
#endif

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.
   This only makes sense on MIPS when using PLTs, so choose the
   PLT relocation (not encountered when not using PLTs).  */
#define ELF_MACHINE_JMP_SLOT			R_MIPS_JUMP_SLOT
#define elf_machine_type_class(type) \
  ((((type) == ELF_MACHINE_JMP_SLOT) * ELF_RTYPE_CLASS_PLT)	\
   | (((type) == R_MIPS_COPY) * ELF_RTYPE_CLASS_COPY))

#define ELF_MACHINE_PLT_REL 1
#define ELF_MACHINE_NO_REL 0
#define ELF_MACHINE_NO_RELA 0

/* Translate a processor specific dynamic tag to the index
   in l_info array.  */
#define DT_MIPS(x) (DT_MIPS_##x - DT_LOPROC + DT_NUM)

/* If there is a DT_MIPS_RLD_MAP_REL or DT_MIPS_RLD_MAP entry in the dynamic
   section, fill in the debug map pointer with the run-time address of the
   r_debug structure.  */
#define ELF_MACHINE_DEBUG_SETUP(l,r) \
do { if ((l)->l_info[DT_MIPS (RLD_MAP_REL)]) \
       { \
	 char *ptr = (char *)(l)->l_info[DT_MIPS (RLD_MAP_REL)]; \
	 ptr += (l)->l_info[DT_MIPS (RLD_MAP_REL)]->d_un.d_val; \
	 *(ElfW(Addr) *)ptr = (ElfW(Addr)) (r); \
       } \
     else if ((l)->l_info[DT_MIPS (RLD_MAP)]) \
       *(ElfW(Addr) *)((l)->l_info[DT_MIPS (RLD_MAP)]->d_un.d_ptr) = \
       (ElfW(Addr)) (r); \
   } while (0)

#if ((defined __mips_nan2008 && !defined HAVE_MIPS_NAN2008) \
     || (!defined __mips_nan2008 && defined HAVE_MIPS_NAN2008))
# error "Configuration inconsistency: __mips_nan2008 != HAVE_MIPS_NAN2008, overridden CFLAGS?"
#endif
#ifdef __mips_nan2008
# define ELF_MACHINE_NAN2008 EF_MIPS_NAN2008
#else
# define ELF_MACHINE_NAN2008 0
#endif

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int __attribute_used__
elf_machine_matches_host (const ElfW(Ehdr) *ehdr)
{
#if _MIPS_SIM == _ABIO32 || _MIPS_SIM == _ABIN32
  /* Don't link o32 and n32 together.  */
  if (((ehdr->e_flags & EF_MIPS_ABI2) != 0) != (_MIPS_SIM == _ABIN32))
    return 0;
#endif

  /* Don't link 2008-NaN and legacy-NaN objects together.  */
  if ((ehdr->e_flags & EF_MIPS_NAN2008) != ELF_MACHINE_NAN2008)
    return 0;

  /* Ensure that the old O32 FP64 ABI is never loaded, it is not supported
     on linux.  */
  if (ehdr->e_flags & EF_MIPS_FP64)
    return 0;

  switch (ehdr->e_machine)
    {
    case EM_MIPS:
    case EM_MIPS_RS3_LE:
      return 1;
    default:
      return 0;
    }
}

static inline ElfW(Addr) *
elf_mips_got_from_gpreg (ElfW(Addr) gpreg)
{
  /* FIXME: the offset of gp from GOT may be system-dependent. */
  return (ElfW(Addr) *) (gpreg - OFFSET_GP_GOT);
}

/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  This must be inlined in a function which
   uses global data.  We assume its $gp points to the primary GOT.  */
static inline ElfW(Addr)
elf_machine_dynamic (void)
{
  register ElfW(Addr) gp __asm__ ("$28");
  return *elf_mips_got_from_gpreg (gp);
}

#define STRINGXP(X) __STRING(X)
#define STRINGXV(X) STRINGV_(X)
#define STRINGV_(...) # __VA_ARGS__

/* Return the run-time load address of the shared object.  */
static inline ElfW(Addr)
elf_machine_load_address (void)
{
  ElfW(Addr) addr;
#ifndef __mips16
  asm ("	.set noreorder\n"
       "	" STRINGXP (PTR_LA) " %0, 0f\n"
# if !defined __mips_isa_rev || __mips_isa_rev < 6
       "	bltzal $0, 0f\n"
       "	nop\n"
       "0:	" STRINGXP (PTR_SUBU) " %0, $31, %0\n"
# else
       "0:	addiupc $31, 0\n"
       "	" STRINGXP (PTR_SUBU) " %0, $31, %0\n"
# endif
       "	.set reorder\n"
       :	"=r" (addr)
       :	/* No inputs */
       :	"$31");
#else
  ElfW(Addr) tmp;
  asm ("	.set noreorder\n"
       "	move %1,$gp\n"
       "	lw %1,%%got(0f)(%1)\n"
       "0:	.fill 0\n"		/* Clear the ISA bit on 0:.  */
       "	la %0,0b\n"
       "	addiu %1,%%lo(0b)\n"
       "	subu %0,%1\n"
       "	.set reorder\n"
       :	"=d" (addr), "=d" (tmp)
       :	/* No inputs */);
#endif
  return addr;
}

/* The MSB of got[1] of a gnu object is set to identify gnu objects.  */
#if _MIPS_SIM == _ABI64
# define ELF_MIPS_GNU_GOT1_MASK	0x8000000000000000L
#else
# define ELF_MIPS_GNU_GOT1_MASK	0x80000000L
#endif

/* We can't rely on elf_machine_got_rel because _dl_object_relocation_scope
   fiddles with global data.  */
#define ELF_MACHINE_BEFORE_RTLD_RELOC(dynamic_info)			\
do {									\
  struct link_map *map = BOOTSTRAP_MAP;					\
  ElfW(Sym) *sym;							\
  ElfW(Addr) *got;							\
  int i, n;								\
									\
  got = (ElfW(Addr) *) D_PTR (map, l_info[DT_PLTGOT]);			\
									\
  if (__builtin_expect (map->l_addr == 0, 1))				\
    break;								\
									\
  /* got[0] is reserved. got[1] is also reserved for the dynamic object	\
     generated by gnu ld. Skip these reserved entries from		\
     relocation.  */							\
  i = (got[1] & ELF_MIPS_GNU_GOT1_MASK)? 2 : 1;				\
  n = map->l_info[DT_MIPS (LOCAL_GOTNO)]->d_un.d_val;			\
									\
  /* Add the run-time displacement to all local got entries. */		\
  while (i < n)								\
    got[i++] += map->l_addr;						\
									\
  /* Handle global got entries. */					\
  got += n;								\
  sym = (ElfW(Sym) *) D_PTR(map, l_info[DT_SYMTAB])			\
       + map->l_info[DT_MIPS (GOTSYM)]->d_un.d_val;			\
  i = (map->l_info[DT_MIPS (SYMTABNO)]->d_un.d_val			\
       - map->l_info[DT_MIPS (GOTSYM)]->d_un.d_val);			\
									\
  while (i--)								\
    {									\
      if (sym->st_shndx == SHN_UNDEF || sym->st_shndx == SHN_COMMON)	\
	*got = SYMBOL_ADDRESS (map, sym, true);				\
      else if (ELFW(ST_TYPE) (sym->st_info) == STT_FUNC			\
	       && *got != sym->st_value)				\
	*got += map->l_addr;						\
      else if (ELFW(ST_TYPE) (sym->st_info) == STT_SECTION)		\
	{								\
	  if (sym->st_other == 0)					\
	    *got += map->l_addr;					\
	}								\
      else								\
	*got = SYMBOL_ADDRESS (map, sym, true);				\
									\
      got++;								\
      sym++;								\
    }									\
} while(0)


/* Mask identifying addresses reserved for the user program,
   where the dynamic linker should not map anything.  */
#define ELF_MACHINE_USER_ADDRESS_MASK	0x80000000UL


/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.
   Note how we have to be careful about two things:

   1) That we allocate a minimal stack of 24 bytes for
      every function call, the MIPS ABI states that even
      if all arguments are passed in registers the procedure
      called can use the 16 byte area pointed to by $sp
      when it is called to store away the arguments passed
      to it.

   2) That under Unix the entry is named __start
      and not just plain _start.  */

#ifndef __mips16
# if !defined __mips_isa_rev || __mips_isa_rev < 6
#  define LCOFF STRINGXP(.Lcof2)
#  define LOAD_31 STRINGXP(bltzal $8) "," STRINGXP(.Lcof2)
# else
#  define LCOFF STRINGXP(.Lcof1)
#  define LOAD_31 "addiupc $31, 0"
# endif
# define RTLD_START asm (\
	".text\n\
	" _RTLD_PROLOGUE(ENTRY_POINT) "\
	" STRINGXV(SETUP_GPX($25)) "\n\
	" STRINGXV(SETUP_GPX64($18,$25)) "\n\
	# i386 ABI book says that the first entry of GOT holds\n\
	# the address of the dynamic structure. Though MIPS ABI\n\
	# doesn't say nothing about this, I emulate this here.\n\
	" STRINGXP(PTR_LA) " $4, _DYNAMIC\n\
	# Subtract OFFSET_GP_GOT\n\
	" STRINGXP(PTR_S) " $4, -0x7ff0($28)\n\
	move $4, $29\n\
	" STRINGXP(PTR_SUBIU) " $29, 16\n\
	\n\
	" STRINGXP(PTR_LA) " $8, " LCOFF "\n\
.Lcof1:	" LOAD_31 "\n\
.Lcof2:	" STRINGXP(PTR_SUBU) " $8, $31, $8\n\
	\n\
	" STRINGXP(PTR_LA) " $25, _dl_start\n\
	" STRINGXP(PTR_ADDU) " $25, $8\n\
	jalr $25\n\
	\n\
	" STRINGXP(PTR_ADDIU) " $29, 16\n\
	# Get the value of label '_dl_start_user' in t9 ($25).\n\
	" STRINGXP(PTR_LA) " $25, _dl_start_user\n\
	" _RTLD_EPILOGUE(ENTRY_POINT) "\
	\n\
	\n\
	" _RTLD_PROLOGUE(_dl_start_user) "\
	" STRINGXP(SETUP_GP) "\n\
	" STRINGXV(SETUP_GP64($18,_dl_start_user)) "\n\
	move $16, $28\n\
	# Save the user entry point address in a saved register.\n\
	move $17, $2\n\
	# See if we were run as a command with the executable file\n\
	# name as an extra leading argument.\n\
	lw $2, _dl_skip_args\n\
	beq $2, $0, 1f\n\
	# Load the original argument count.\n\
	" STRINGXP(PTR_L) " $4, 0($29)\n\
	# Subtract _dl_skip_args from it.\n\
	subu $4, $2\n\
	# Adjust the stack pointer to skip _dl_skip_args words.\n\
	sll $2, " STRINGXP (PTRLOG) "\n\
	" STRINGXP(PTR_ADDU) " $29, $2\n\
	# Save back the modified argument count.\n\
	" STRINGXP(PTR_S) " $4, 0($29)\n\
1:	# Call _dl_init (struct link_map *main_map, int argc, char **argv, char **env) \n\
	" STRINGXP(PTR_L) " $4, _rtld_local\n\
	" STRINGXP(PTR_L) /* or lw???  fixme */ " $5, 0($29)\n\
	" STRINGXP(PTR_LA) " $6, " STRINGXP (PTRSIZE) "($29)\n\
	sll $7, $5, " STRINGXP (PTRLOG) "\n\
	" STRINGXP(PTR_ADDU) " $7, $7, $6\n\
	" STRINGXP(PTR_ADDU) " $7, $7, " STRINGXP (PTRSIZE) " \n\
	# Make sure the stack pointer is aligned for _dl_init.\n\
	and $2, $29, -2 * " STRINGXP(SZREG) "\n\
	move $8, $29\n\
	" STRINGXP(PTR_SUBIU) " $29, $2, 32\n\
	" STRINGXP(PTR_S) " $8, (32 - " STRINGXP(SZREG) ")($29)\n\
	" STRINGXP(SAVE_GP(16)) "\n\
	# Call the function to run the initializers.\n\
	jal _dl_init\n\
	# Restore the stack pointer for _start.\n\
	" STRINGXP(PTR_L)  " $29, (32 - " STRINGXP(SZREG) ")($29)\n\
	# Pass our finalizer function to the user in $2 as per ELF ABI.\n\
	" STRINGXP(PTR_LA) " $2, _dl_fini\n\
	# Jump to the user entry point.\n\
	move $25, $17\n\
	jr $25\n\t"\
	_RTLD_EPILOGUE(_dl_start_user)\
	".previous"\
);

#else /* __mips16 */
/* MIPS16 version.  We currently only support O32 under MIPS16; the proper
   assembly preprocessor abstractions will need to be added if other ABIs
   are to be supported.  */

# define RTLD_START asm (\
	".text\n\
	.set mips16\n\
	" _RTLD_PROLOGUE (ENTRY_POINT) "\
	# Construct GP value in $3.\n\
	li $3, %hi(_gp_disp)\n\
	addiu $4, $pc, %lo(_gp_disp)\n\
	sll $3, 16\n\
	addu $3, $4\n\
	move $28, $3\n\
	lw $4, %got(_DYNAMIC)($3)\n\
	sw $4, -0x7ff0($3)\n\
	move $4, $sp\n\
	addiu $sp, -16\n\
	# _dl_start() is sufficiently near to use pc-relative\n\
	# load address.\n\
	la $3, _dl_start\n\
	move $25, $3\n\
	jalr $3\n\
	addiu $sp, 16\n\
	" _RTLD_EPILOGUE (ENTRY_POINT) "\
	\n\
	\n\
	" _RTLD_PROLOGUE (_dl_start_user) "\
	li $16, %hi(_gp_disp)\n\
	addiu $4, $pc, %lo(_gp_disp)\n\
	sll $16, 16\n\
	addu $16, $4\n\
	move $17, $2\n\
	move $28, $16\n\
	lw $4, %got(_dl_skip_args)($16)\n\
	lw $4, 0($4)\n\
	beqz $4, 1f\n\
	# Load the original argument count.\n\
	lw $5, 0($sp)\n\
	# Subtract _dl_skip_args from it.\n\
	subu $5, $4\n\
	# Adjust the stack pointer to skip _dl_skip_args words.\n\
	sll $4, " STRINGXP (PTRLOG) "\n\
	move $6, $sp\n\
	addu $6, $4\n\
	move $sp, $6\n\
	# Save back the modified argument count.\n\
	sw $5, 0($sp)\n\
1:	# Call _dl_init (struct link_map *main_map, int argc, char **argv, char **env) \n\
	lw $4, %got(_rtld_local)($16)\n\
	lw $4, 0($4)\n\
	lw $5, 0($sp)\n\
	addiu $6, $sp, " STRINGXP (PTRSIZE) "\n\
	sll $7, $5, " STRINGXP (PTRLOG) "\n\
	addu $7, $6\n\
	addu $7, " STRINGXP (PTRSIZE) "\n\
	# Make sure the stack pointer is aligned for _dl_init.\n\
	li $2, 2 * " STRINGXP (SZREG) "\n\
	neg $2, $2\n\
	move $3, $sp\n\
	and $2, $3\n\
	sw $3, -" STRINGXP (SZREG) "($2)\n\
	addiu $2, -32\n\
	move $sp, $2\n\
	sw $16, 16($sp)\n\
	# Call the function to run the initializers.\n\
	lw $2, %call16(_dl_init)($16)\n\
	move $25, $2\n\
	jalr $2\n\
	# Restore the stack pointer for _start.\n\
	lw $2, 32-" STRINGXP (SZREG) "($sp)\n\
	move $sp, $2\n\
	move $28, $16\n\
	# Pass our finalizer function to the user in $2 as per ELF ABI.\n\
	lw $2, %call16(_dl_fini)($16)\n\
	# Jump to the user entry point.\n\
	move $25, $17\n\
	jr $17\n\t"\
	_RTLD_EPILOGUE (_dl_start_user)\
	".previous"\
);

#endif /* __mips16 */

/* Names of the architecture-specific auditing callback functions.  */
# if _MIPS_SIM == _ABIO32
#  define ARCH_LA_PLTENTER mips_o32_gnu_pltenter
#  define ARCH_LA_PLTEXIT mips_o32_gnu_pltexit
# elif _MIPS_SIM == _ABIN32
#  define ARCH_LA_PLTENTER mips_n32_gnu_pltenter
#  define ARCH_LA_PLTEXIT mips_n32_gnu_pltexit
# else
#  define ARCH_LA_PLTENTER mips_n64_gnu_pltenter
#  define ARCH_LA_PLTEXIT mips_n64_gnu_pltexit
# endif

/* We define an initialization function.  This is called very early in
   _dl_sysdep_start.  */
#define DL_PLATFORM_INIT dl_platform_init ()

static inline void __attribute__ ((unused))
dl_platform_init (void)
{
  if (GLRO(dl_platform) != NULL && *GLRO(dl_platform) == '\0')
    /* Avoid an empty string which would disturb us.  */
    GLRO(dl_platform) = NULL;
}

/* For a non-writable PLT, rewrite the .got.plt entry at RELOC_ADDR to
   point at the symbol with address VALUE.  For a writable PLT, rewrite
   the corresponding PLT entry instead.  */
static inline ElfW(Addr)
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const ElfW(Rel) *reloc,
		       ElfW(Addr) *reloc_addr, ElfW(Addr) value)
{
  return *reloc_addr = value;
}

static inline ElfW(Addr)
elf_machine_plt_value (struct link_map *map, const ElfW(Rel) *reloc,
		       ElfW(Addr) value)
{
  return value;
}

#endif /* !dl_machine_h */

#ifdef RESOLVE_MAP

/* Perform a relocation described by R_INFO at the location pointed to
   by RELOC_ADDR.  SYM is the relocation symbol specified by R_INFO and
   MAP is the object containing the reloc.  */

auto inline void
__attribute__ ((always_inline))
elf_machine_reloc (struct link_map *map, ElfW(Addr) r_info,
		   const ElfW(Sym) *sym, const struct r_found_version *version,
		   void *reloc_addr, ElfW(Addr) r_addend, int inplace_p)
{
  const unsigned long int r_type = ELFW(R_TYPE) (r_info);
  ElfW(Addr) *addr_field = (ElfW(Addr) *) reloc_addr;

#if !defined RTLD_BOOTSTRAP && !defined SHARED
  /* This is defined in rtld.c, but nowhere in the static libc.a;
     make the reference weak so static programs can still link.  This
     declaration cannot be done when compiling rtld.c (i.e.  #ifdef
     RTLD_BOOTSTRAP) because rtld.c contains the common defn for
     _dl_rtld_map, which is incompatible with a weak decl in the same
     file.  */
  weak_extern (GL(dl_rtld_map));
#endif

  switch (r_type)
    {
#if !defined (RTLD_BOOTSTRAP)
# if _MIPS_SIM == _ABI64
    case R_MIPS_TLS_DTPMOD64:
    case R_MIPS_TLS_DTPREL64:
    case R_MIPS_TLS_TPREL64:
# else
    case R_MIPS_TLS_DTPMOD32:
    case R_MIPS_TLS_DTPREL32:
    case R_MIPS_TLS_TPREL32:
# endif
      {
	struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);

	switch (r_type)
	  {
	  case R_MIPS_TLS_DTPMOD64:
	  case R_MIPS_TLS_DTPMOD32:
	    if (sym_map)
	      *addr_field = sym_map->l_tls_modid;
	    break;

	  case R_MIPS_TLS_DTPREL64:
	  case R_MIPS_TLS_DTPREL32:
	    if (sym)
	      {
		if (inplace_p)
		  r_addend = *addr_field;
		*addr_field = r_addend + TLS_DTPREL_VALUE (sym);
	      }
	    break;

	  case R_MIPS_TLS_TPREL32:
	  case R_MIPS_TLS_TPREL64:
	    if (sym)
	      {
		CHECK_STATIC_TLS (map, sym_map);
		if (inplace_p)
		  r_addend = *addr_field;
		*addr_field = r_addend + TLS_TPREL_VALUE (sym_map, sym);
	      }
	    break;
	  }

	break;
      }
#endif

#if _MIPS_SIM == _ABI64
    case (R_MIPS_64 << 8) | R_MIPS_REL32:
#else
    case R_MIPS_REL32:
#endif
      {
	int symidx = ELFW(R_SYM) (r_info);
	ElfW(Addr) reloc_value;

	if (inplace_p)
	  /* Support relocations on mis-aligned offsets.  */
	  __builtin_memcpy (&reloc_value, reloc_addr, sizeof (reloc_value));
	else
	  reloc_value = r_addend;

	if (symidx)
	  {
	    const ElfW(Word) gotsym
	      = (const ElfW(Word)) map->l_info[DT_MIPS (GOTSYM)]->d_un.d_val;

	    if ((ElfW(Word))symidx < gotsym)
	      {
		/* This wouldn't work for a symbol imported from other
		   libraries for which there's no GOT entry, but MIPS
		   requires every symbol referenced in a dynamic
		   relocation to have a GOT entry in the primary GOT,
		   so we only get here for locally-defined symbols.
		   For section symbols, we should *NOT* be adding
		   sym->st_value (per the definition of the meaning of
		   S in reloc expressions in the ELF64 MIPS ABI),
		   since it should have already been added to
		   reloc_value by the linker, but older versions of
		   GNU ld didn't add it, and newer versions don't emit
		   useless relocations to section symbols any more, so
		   it is safe to keep on adding sym->st_value, even
		   though it's not ABI compliant.  Some day we should
		   bite the bullet and stop doing this.  */
#ifndef RTLD_BOOTSTRAP
		if (map != &GL(dl_rtld_map))
#endif
		  reloc_value += SYMBOL_ADDRESS (map, sym, true);
	      }
	    else
	      {
#ifndef RTLD_BOOTSTRAP
		const ElfW(Addr) *got
		  = (const ElfW(Addr) *) D_PTR (map, l_info[DT_PLTGOT]);
		const ElfW(Word) local_gotno
		  = (const ElfW(Word))
		    map->l_info[DT_MIPS (LOCAL_GOTNO)]->d_un.d_val;

		reloc_value += got[symidx + local_gotno - gotsym];
#endif
	      }
	  }
	else
#ifndef RTLD_BOOTSTRAP
	  if (map != &GL(dl_rtld_map))
#endif
	    reloc_value += map->l_addr;

	__builtin_memcpy (reloc_addr, &reloc_value, sizeof (reloc_value));
      }
      break;
#ifndef RTLD_BOOTSTRAP
#if _MIPS_SIM == _ABI64
    case (R_MIPS_64 << 8) | R_MIPS_GLOB_DAT:
#else
    case R_MIPS_GLOB_DAT:
#endif
      {
	int symidx = ELFW(R_SYM) (r_info);
	const ElfW(Word) gotsym
	  = (const ElfW(Word)) map->l_info[DT_MIPS (GOTSYM)]->d_un.d_val;

	if (__builtin_expect ((ElfW(Word)) symidx >= gotsym, 1))
	  {
	    const ElfW(Addr) *got
	      = (const ElfW(Addr) *) D_PTR (map, l_info[DT_PLTGOT]);
	    const ElfW(Word) local_gotno
	      = ((const ElfW(Word))
		 map->l_info[DT_MIPS (LOCAL_GOTNO)]->d_un.d_val);

	    ElfW(Addr) reloc_value = got[symidx + local_gotno - gotsym];
	    __builtin_memcpy (reloc_addr, &reloc_value, sizeof (reloc_value));
	  }
      }
      break;
#endif
    case R_MIPS_NONE:		/* Alright, Wilbur.  */
      break;

    case R_MIPS_JUMP_SLOT:
      {
	struct link_map *sym_map;
	ElfW(Addr) value;

	/* The addend for a jump slot relocation must always be zero:
	   calls via the PLT always branch to the symbol's address and
	   not to the address plus a non-zero offset.  */
	if (r_addend != 0)
	  _dl_signal_error (0, map->l_name, NULL,
			    "found jump slot relocation with non-zero addend");

	sym_map = RESOLVE_MAP (&sym, version, r_type);
	value = SYMBOL_ADDRESS (sym_map, sym, true);
	*addr_field = value;

	break;
      }

    case R_MIPS_COPY:
      {
	const ElfW(Sym) *const refsym = sym;
	struct link_map *sym_map;
	ElfW(Addr) value;

	/* Calculate the address of the symbol.  */
	sym_map = RESOLVE_MAP (&sym, version, r_type);
	value = SYMBOL_ADDRESS (sym_map, sym, true);

	if (__builtin_expect (sym == NULL, 0))
	  /* This can happen in trace mode if an object could not be
	     found.  */
	  break;
	if (__builtin_expect (sym->st_size > refsym->st_size, 0)
	    || (__builtin_expect (sym->st_size < refsym->st_size, 0)
		&& GLRO(dl_verbose)))
	  {
	    const char *strtab;

	    strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
	    _dl_error_printf ("\
  %s: Symbol `%s' has different size in shared object, consider re-linking\n",
			      RTLD_PROGNAME, strtab + refsym->st_name);
	  }
	memcpy (reloc_addr, (void *) value,
		sym->st_size < refsym->st_size
		? sym->st_size : refsym->st_size);
	break;
      }

#if _MIPS_SIM == _ABI64
    case R_MIPS_64:
      /* For full compliance with the ELF64 ABI, one must precede the
	 _REL32/_64 pair of relocations with a _64 relocation, such
	 that the in-place addend is read as a 64-bit value.  IRIX
	 didn't pick up on this requirement, so we treat the
	 _REL32/_64 relocation as a 64-bit relocation even if it's by
	 itself.  For ABI compliance, we ignore such _64 dummy
	 relocations.  For RELA, this may be simply removed, since
	 it's totally unnecessary.  */
      if (ELFW(R_SYM) (r_info) == 0)
	break;
#endif
      /* Fall through.  */
    default:
      _dl_reloc_bad_type (map, r_type, 0);
      break;
    }
}

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   MAP is the object containing the reloc.  */

auto inline void
__attribute__ ((always_inline))
elf_machine_rel (struct link_map *map, const ElfW(Rel) *reloc,
		 const ElfW(Sym) *sym, const struct r_found_version *version,
		 void *const reloc_addr, int skip_ifunc)
{
  elf_machine_reloc (map, reloc->r_info, sym, version, reloc_addr, 0, 1);
}

auto inline void
__attribute__((always_inline))
elf_machine_rel_relative (ElfW(Addr) l_addr, const ElfW(Rel) *reloc,
			  void *const reloc_addr)
{
  /* XXX Nothing to do.  There is no relative relocation, right?  */
}

auto inline void
__attribute__((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      ElfW(Addr) l_addr, const ElfW(Rel) *reloc,
		      int skip_ifunc)
{
  ElfW(Addr) *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  const unsigned int r_type = ELFW(R_TYPE) (reloc->r_info);
  /* Check for unexpected PLT reloc type.  */
  if (__builtin_expect (r_type == R_MIPS_JUMP_SLOT, 1))
    {
      if (__builtin_expect (map->l_mach.plt, 0) == 0)
	{
	  /* Nothing is required here since we only support lazy
	     relocation in executables.  */
	}
      else
	*reloc_addr = map->l_mach.plt;
    }
  else
    _dl_reloc_bad_type (map, r_type, 1);
}

auto inline void
__attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const ElfW(Rela) *reloc,
		  const ElfW(Sym) *sym, const struct r_found_version *version,
		  void *const reloc_addr, int skip_ifunc)
{
  elf_machine_reloc (map, reloc->r_info, sym, version, reloc_addr,
		     reloc->r_addend, 0);
}

auto inline void
__attribute__((always_inline))
elf_machine_rela_relative (ElfW(Addr) l_addr, const ElfW(Rela) *reloc,
			   void *const reloc_addr)
{
}

#ifndef RTLD_BOOTSTRAP
/* Relocate GOT. */
auto inline void
__attribute__((always_inline))
elf_machine_got_rel (struct link_map *map, int lazy)
{
  ElfW(Addr) *got;
  ElfW(Sym) *sym;
  const ElfW(Half) *vernum;
  int i, n, symidx;

#define RESOLVE_GOTSYM(sym,vernum,sym_index,reloc)			  \
    ({									  \
      const ElfW(Sym) *ref = sym;					  \
      const struct r_found_version *version __attribute__ ((unused))	  \
	= vernum ? &map->l_versions[vernum[sym_index] & 0x7fff] : NULL;	  \
      struct link_map *sym_map;						  \
      sym_map = RESOLVE_MAP (&ref, version, reloc);			  \
      SYMBOL_ADDRESS (sym_map, ref, true);				  \
    })

  if (map->l_info[VERSYMIDX (DT_VERSYM)] != NULL)
    vernum = (const void *) D_PTR (map, l_info[VERSYMIDX (DT_VERSYM)]);
  else
    vernum = NULL;

  got = (ElfW(Addr) *) D_PTR (map, l_info[DT_PLTGOT]);

  n = map->l_info[DT_MIPS (LOCAL_GOTNO)]->d_un.d_val;
  /* The dynamic linker's local got entries have already been relocated.  */
  if (map != &GL(dl_rtld_map))
    {
      /* got[0] is reserved. got[1] is also reserved for the dynamic object
	 generated by gnu ld. Skip these reserved entries from relocation.  */
      i = (got[1] & ELF_MIPS_GNU_GOT1_MASK)? 2 : 1;

      /* Add the run-time displacement to all local got entries if
	 needed.  */
      if (__builtin_expect (map->l_addr != 0, 0))
	{
	  while (i < n)
	    got[i++] += map->l_addr;
	}
    }

  /* Handle global got entries. */
  got += n;
  /* Keep track of the symbol index.  */
  symidx = map->l_info[DT_MIPS (GOTSYM)]->d_un.d_val;
  sym = (ElfW(Sym) *) D_PTR (map, l_info[DT_SYMTAB]) + symidx;
  i = (map->l_info[DT_MIPS (SYMTABNO)]->d_un.d_val
       - map->l_info[DT_MIPS (GOTSYM)]->d_un.d_val);

  /* This loop doesn't handle Quickstart.  */
  while (i--)
    {
      if (sym->st_shndx == SHN_UNDEF)
	{
	  if (ELFW(ST_TYPE) (sym->st_info) == STT_FUNC && sym->st_value
	      && !(sym->st_other & STO_MIPS_PLT))
	    {
	      if (lazy)
		*got = SYMBOL_ADDRESS (map, sym, true);
	      else
		/* This is a lazy-binding stub, so we don't need the
		   canonical address.  */
		*got = RESOLVE_GOTSYM (sym, vernum, symidx, R_MIPS_JUMP_SLOT);
	    }
	  else
	    *got = RESOLVE_GOTSYM (sym, vernum, symidx, R_MIPS_32);
	}
      else if (sym->st_shndx == SHN_COMMON)
	*got = RESOLVE_GOTSYM (sym, vernum, symidx, R_MIPS_32);
      else if (ELFW(ST_TYPE) (sym->st_info) == STT_FUNC
	       && *got != sym->st_value)
	{
	  if (lazy)
	    *got += map->l_addr;
	  else
	    /* This is a lazy-binding stub, so we don't need the
	       canonical address.  */
	    *got = RESOLVE_GOTSYM (sym, vernum, symidx, R_MIPS_JUMP_SLOT);
	}
      else if (ELFW(ST_TYPE) (sym->st_info) == STT_SECTION)
	{
	  if (sym->st_other == 0)
	    *got += map->l_addr;
	}
      else
	*got = RESOLVE_GOTSYM (sym, vernum, symidx, R_MIPS_32);

      ++got;
      ++sym;
      ++symidx;
    }

#undef RESOLVE_GOTSYM
}
#endif

/* Set up the loaded object described by L so its stub function
   will jump to the on-demand fixup code __dl_runtime_resolve.  */

auto inline int
__attribute__((always_inline))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
# ifndef RTLD_BOOTSTRAP
  ElfW(Addr) *got;
  extern void _dl_runtime_resolve (ElfW(Word));
  extern void _dl_runtime_pltresolve (void);
  extern int _dl_mips_gnu_objects;

  if (lazy)
    {
      /* The GOT entries for functions have not yet been filled in.
	 Their initial contents will arrange when called to put an
	 offset into the .dynsym section in t8, the return address
	 in t7 and then jump to _GLOBAL_OFFSET_TABLE[0].  */
      got = (ElfW(Addr) *) D_PTR (l, l_info[DT_PLTGOT]);

      /* This function will get called to fix up the GOT entry indicated by
	 the register t8, and then jump to the resolved address.  */
      got[0] = (ElfW(Addr)) &_dl_runtime_resolve;

      /* Store l to _GLOBAL_OFFSET_TABLE[1] for gnu object. The MSB
	 of got[1] of a gnu object is set to identify gnu objects.
	 Where we can store l for non gnu objects? XXX  */
      if ((got[1] & ELF_MIPS_GNU_GOT1_MASK) != 0)
	got[1] = ((ElfW(Addr)) l | ELF_MIPS_GNU_GOT1_MASK);
      else
	_dl_mips_gnu_objects = 0;
    }

  /* Relocate global offset table.  */
  elf_machine_got_rel (l, lazy);

  /* If using PLTs, fill in the first two entries of .got.plt.  */
  if (l->l_info[DT_JMPREL] && lazy)
    {
      ElfW(Addr) *gotplt;
      gotplt = (ElfW(Addr) *) D_PTR (l, l_info[DT_MIPS (PLTGOT)]);
      /* If a library is prelinked but we have to relocate anyway,
	 we have to be able to undo the prelinking of .got.plt.
	 The prelinker saved the address of .plt for us here.  */
      if (gotplt[1])
	l->l_mach.plt = gotplt[1] + l->l_addr;
      gotplt[0] = (ElfW(Addr)) &_dl_runtime_pltresolve;
      gotplt[1] = (ElfW(Addr)) l;
    }

# endif
  return lazy;
}

#endif /* RESOLVE_MAP */
