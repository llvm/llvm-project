/* Copyright (C) 2005-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _LINK_H
# error "Never include <bits/link.h> directly; use <link.h> instead."
#endif

/* Registers for entry into PLT on Microblaze.  */
typedef struct La_microblaze_regs
{
  uint32_t lr_r0;
  uint32_t lr_r1;
  uint32_t lr_sp;
} La_microblaze_regs;

/* Return values for calls from PLT on Microblaze.  */
typedef struct La_microblaze_retval
{
  uint32_t lrv_r3;
} La_microblaze_retval;


__BEGIN_DECLS

extern Elf32_Addr la_microblaze_gnu_pltenter (Elf32_Sym *__sym,
                                              unsigned int __ndx,
                                              uintptr_t *__refcook,
                                              uintptr_t *__defcook,
                                              La_microblaze_regs *__regs,
                                              unsigned int *__flags,
                                              const char *__symname,
                                              long int *__framesizep);

extern unsigned int la_microblaze_gnu_pltexit (Elf32_Sym *__sym,
                                               unsigned int __ndx,
                                               uintptr_t *__refcook,
                                               uintptr_t *__defcook,
                                               const La_microblaze_regs *__inregs,
                                               La_microblaze_retval *__outregs,
                                               const char *__symname);

__END_DECLS
