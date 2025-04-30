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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef	_LINK_H
# error "Never include <bits/link.h> directly; use <link.h> instead."
#endif


/* Registers for entry into PLT on Alpha.  */
typedef struct La_alpha_regs
{
  uint64_t lr_r26;
  uint64_t lr_sp;
  uint64_t lr_r16;
  uint64_t lr_r17;
  uint64_t lr_r18;
  uint64_t lr_r19;
  uint64_t lr_r20;
  uint64_t lr_r21;
  double lr_f16;
  double lr_f17;
  double lr_f18;
  double lr_f19;
  double lr_f20;
  double lr_f21;
} La_alpha_regs;

/* Return values for calls from PLT on Alpha.  */
typedef struct La_alpha_retval
{
  uint64_t lrv_r0;
  uint64_t lrv_r1;
  double lrv_f0;
  double lrv_f1;
} La_alpha_retval;


__BEGIN_DECLS

extern Elf64_Addr la_alpha_gnu_pltenter (Elf64_Sym *__sym, unsigned int __ndx,
				         uintptr_t *__refcook,
				         uintptr_t *__defcook,
				         La_alpha_regs *__regs,
				         unsigned int *__flags,
				         const char *__symname,
				         long int *__framesizep);
extern unsigned int la_alpha_gnu_pltexit (Elf64_Sym *__sym, unsigned int __ndx,
					  uintptr_t *__refcook,
					  uintptr_t *__defcook,
					  const La_alpha_regs *__inregs,
					  La_alpha_retval *__outregs,
					  const char *symname);

__END_DECLS
