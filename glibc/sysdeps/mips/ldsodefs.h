/* Run-time dynamic linker data structures for loaded ELF shared objects.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef _MIPS_LDSODEFS_H
#define _MIPS_LDSODEFS_H 1

#include <elf.h>

struct La_mips_32_regs;
struct La_mips_32_retval;
struct La_mips_64_regs;
struct La_mips_64_retval;

#define ELF_MACHINE_GNU_HASH_ADDRIDX (DT_MIPS_XHASH - DT_LOPROC + DT_NUM)

/* Calculate the index of a symbol in MIPS xhash.  */
#define ELF_MACHINE_HASH_SYMIDX(map, hasharr) \
  ((map)->l_mach.mips_xlat_zero[(hasharr) - (map)->l_gnu_chain_zero])

/* Setup MIPS xhash.  */
#define ELF_MACHINE_XHASH_SETUP(hash32, symbias, map)			    \
  do									    \
    {									    \
      (hash32) += (map)->l_info[DT_MIPS (SYMTABNO)]->d_un.d_val - (symbias); \
      (map)->l_mach.mips_xlat_zero = (hash32) - (symbias);		    \
    }									    \
  while (0)

#define ARCH_PLTENTER_MEMBERS						    \
    Elf32_Addr (*mips_o32_gnu_pltenter) (Elf32_Sym *, unsigned int,	    \
					 uintptr_t *, uintptr_t *,	    \
					 struct La_mips_32_regs *,	    \
					 unsigned int *, const char *name,  \
					 long int *framesizep);		    \
    Elf32_Addr (*mips_n32_gnu_pltenter) (Elf32_Sym *, unsigned int,	    \
					 uintptr_t *, uintptr_t *,	    \
					 struct La_mips_64_regs *,	    \
					 unsigned int *, const char *name,  \
					 long int *framesizep);		    \
    Elf64_Addr (*mips_n64_gnu_pltenter) (Elf64_Sym *, unsigned int,	    \
					 uintptr_t *, uintptr_t *,	    \
					 struct La_mips_64_regs *,	    \
					 unsigned int *, const char *name,  \
					 long int *framesizep);

#define ARCH_PLTEXIT_MEMBERS						    \
    unsigned int (*mips_o32_gnu_pltexit) (Elf32_Sym *, unsigned int,	    \
					  uintptr_t *, uintptr_t *,	    \
					  const struct La_mips_32_regs *,   \
					  struct La_mips_32_retval *,	    \
					  const char *);		    \
    unsigned int (*mips_n32_gnu_pltexit) (Elf32_Sym *, unsigned int,	    \
					  uintptr_t *, uintptr_t *,	    \
					  const struct La_mips_64_regs *,   \
					  struct La_mips_64_retval *,	    \
					  const char *);		    \
    unsigned int (*mips_n64_gnu_pltexit) (Elf64_Sym *, unsigned int,	    \
					  uintptr_t *, uintptr_t *,	    \
					  const struct La_mips_64_regs *,   \
					  struct La_mips_64_retval *,	    \
					  const char *);

/* The MIPS ABI specifies that the dynamic section has to be read-only.  */

#define DL_RO_DYN_SECTION 1

#include_next <ldsodefs.h>

/* The 64-bit MIPS ELF ABI uses an unusual reloc format.  Each
   relocation entry specifies up to three actual relocations, all at
   the same address.  The first relocation which required a symbol
   uses the symbol in the r_sym field.  The second relocation which
   requires a symbol uses the symbol in the r_ssym field.  If all
   three relocations require a symbol, the third one uses a zero
   value.

   We define these structures in internal headers because we're not
   sure we want to make them part of the ABI yet.  Eventually, some of
   this may move into elf/elf.h.  */

/* An entry in a 64 bit SHT_REL section.  */

typedef struct
{
  Elf32_Word    r_sym;		/* Symbol index */
  unsigned char r_ssym;		/* Special symbol for 2nd relocation */
  unsigned char r_type3;	/* 3rd relocation type */
  unsigned char r_type2;	/* 2nd relocation type */
  unsigned char r_type1;	/* 1st relocation type */
} _Elf64_Mips_R_Info;

typedef union
{
  Elf64_Xword	r_info_number;
  _Elf64_Mips_R_Info r_info_fields;
} _Elf64_Mips_R_Info_union;

typedef struct
{
  Elf64_Addr	r_offset;		/* Address */
  _Elf64_Mips_R_Info_union r_info;	/* Relocation type and symbol index */
} Elf64_Mips_Rel;

typedef struct
{
  Elf64_Addr	r_offset;		/* Address */
  _Elf64_Mips_R_Info_union r_info;	/* Relocation type and symbol index */
  Elf64_Sxword	r_addend;		/* Addend */
} Elf64_Mips_Rela;

#define ELF64_MIPS_R_SYM(i) \
  ((__extension__ (_Elf64_Mips_R_Info_union)(i)).r_info_fields.r_sym)
#define ELF64_MIPS_R_TYPE(i) \
  (((_Elf64_Mips_R_Info_union)(i)).r_info_fields.r_type1 \
   | ((Elf32_Word)(__extension__ (_Elf64_Mips_R_Info_union)(i) \
		   ).r_info_fields.r_type2 << 8) \
   | ((Elf32_Word)(__extension__ (_Elf64_Mips_R_Info_union)(i) \
		   ).r_info_fields.r_type3 << 16) \
   | ((Elf32_Word)(__extension__ (_Elf64_Mips_R_Info_union)(i) \
		   ).r_info_fields.r_ssym << 24))
#define ELF64_MIPS_R_INFO(sym, type) \
  (__extension__ (_Elf64_Mips_R_Info_union) \
   (__extension__ (_Elf64_Mips_R_Info) \
   { (sym), ELF64_MIPS_R_SSYM (type), \
       ELF64_MIPS_R_TYPE3 (type), \
       ELF64_MIPS_R_TYPE2 (type), \
       ELF64_MIPS_R_TYPE1 (type) \
   }).r_info_number)
/* These macros decompose the value returned by ELF64_MIPS_R_TYPE, and
   compose it back into a value that it can be used as an argument to
   ELF64_MIPS_R_INFO.  */
#define ELF64_MIPS_R_SSYM(i) (((i) >> 24) & 0xff)
#define ELF64_MIPS_R_TYPE3(i) (((i) >> 16) & 0xff)
#define ELF64_MIPS_R_TYPE2(i) (((i) >> 8) & 0xff)
#define ELF64_MIPS_R_TYPE1(i) ((i) & 0xff)
#define ELF64_MIPS_R_TYPEENC(type1, type2, type3, ssym) \
  ((type1) \
   | ((Elf32_Word)(type2) << 8) \
   | ((Elf32_Word)(type3) << 16) \
   | ((Elf32_Word)(ssym) << 24))

#undef ELF64_R_SYM
#define ELF64_R_SYM(i) ELF64_MIPS_R_SYM (i)
#undef ELF64_R_TYPE
#define ELF64_R_TYPE(i) ELF64_MIPS_R_TYPE (i)
#undef ELF64_R_INFO
#define ELF64_R_INFO(sym, type) ELF64_MIPS_R_INFO ((sym), (type))

#endif
