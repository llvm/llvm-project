/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#ifndef	_LINK_H
# error "Never include <bits/link.h> directly; use <link.h> instead."
#endif

/* Registers for entry into PLT on ia64.  */
typedef struct La_ia64_regs
{
  uint64_t lr_r8;
  uint64_t lr_r9;
  uint64_t lr_r10;
  uint64_t lr_r11;
  uint64_t lr_gr [8];
  long double lr_fr [8];
  uint64_t lr_unat;
  uint64_t lr_sp;
} La_ia64_regs;

/* Return values for calls from PLT on ia64.  */
typedef struct La_ia64_retval
{
  uint64_t lrv_r8;
  uint64_t lrv_r9;
  uint64_t lrv_r10;
  uint64_t lrv_r11;
  long double lr_fr [8];
} La_ia64_retval;


__BEGIN_DECLS

extern Elf64_Addr la_ia64_gnu_pltenter (Elf64_Sym *__sym, unsigned int __ndx,
					uintptr_t *__refcook,
					uintptr_t *__defcook,
					La_ia64_regs *__regs,
					unsigned int *__flags,
					const char *__symname,
					long int *__framesizep);
extern unsigned int la_ia64_gnu_pltexit (Elf64_Sym *__sym, unsigned int __ndx,
					 uintptr_t *__refcook,
					 uintptr_t *__defcook,
					 const La_ia64_regs *__inregs,
					 La_ia64_retval *__outregs,
					 const char *__symname);

__END_DECLS
