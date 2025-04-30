/* Run-time dynamic linker data structures for loaded ELF shared objects.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

#ifndef _RISCV_LDSODEFS_H
#define _RISCV_LDSODEFS_H 1

#include <elf.h>

struct La_riscv_regs;
struct La_riscv_retval;

#define ARCH_PLTENTER_MEMBERS						\
    ElfW(Addr) (*riscv_gnu_pltenter) (ElfW(Sym) *, unsigned int,	\
				      uintptr_t *, uintptr_t *,		\
				      const struct La_riscv_regs *,	\
				      unsigned int *, const char *name,	\
				      long int *framesizep);

#define ARCH_PLTEXIT_MEMBERS						\
    unsigned int (*riscv_gnu_pltexit) (ElfW(Sym) *, unsigned int,	\
				       uintptr_t *, uintptr_t *,	\
				       const struct La_riscv_regs *,	\
				       struct La_riscv_retval *,	\
				       const char *);

/* Although the RISC-V ABI does not specify that the dynamic section has
   to be read-only, it needs to be kept for ABI compatibility.  */

#define DL_RO_DYN_SECTION 1

#include_next <ldsodefs.h>

#endif
