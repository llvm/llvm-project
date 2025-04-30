/* Run-time dynamic linker data structures for loaded ELF shared objects.
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

#ifndef	_SH_LDSODEFS_H
#define	_SH_LDSODEFS_H	1

#include <elf.h>

struct La_sh_regs;
struct La_sh_retval;

#define ARCH_PLTENTER_MEMBERS						\
    uintptr_t (*sh_gnu_pltenter) (Elf32_Sym *, unsigned int, uintptr_t *, \
				  uintptr_t *, struct La_sh_regs *, \
				  unsigned int *, const char *name,	\
				  long int *framesizep)

#define ARCH_PLTEXIT_MEMBERS						\
    unsigned int (*sh_gnu_pltexit) (Elf32_Sym *, unsigned int, uintptr_t *, \
				    uintptr_t *, const struct La_sh_regs *, \
				    struct La_sh_retval *, const char *)

#include_next <ldsodefs.h>

#endif
