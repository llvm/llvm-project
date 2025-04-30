/* Function descriptors.  HPPA version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#ifndef dl_hppa_fptr_h
#define dl_hppa_fptr_h 1

#include <sysdeps/generic/dl-fptr.h>

/* Initialize function pointer code. Call before relocation processing.  */
extern void _dl_fptr_init (void);

/* There are currently 33 dynamic symbols in ld.so.
   ELF_MACHINE_BOOT_FPTR_TABLE_LEN needs to be at least that big.  */
#define ELF_MACHINE_BOOT_FPTR_TABLE_LEN 64

#define ELF_MACHINE_LOAD_ADDRESS(var, symbol) \
  asm (								\
"	b,l	1f,%0\n"					\
"	addil	L'" #symbol " - ($PIC_pcrel$0 - 1),%0\n"	\
"1:	ldo	R'" #symbol " - ($PIC_pcrel$0 - 5)(%%r1),%0\n"	\
   : "=&r" (var) : : "r1");

#endif /* !dl_hppa_fptr_h */
