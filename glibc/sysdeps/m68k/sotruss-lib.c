/* Override generic sotruss-lib.c to define actual functions for m68k.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#define HAVE_ARCH_PLTENTER
#define HAVE_ARCH_PLTEXIT

#include <elf/sotruss-lib.c>

ElfW(Addr)
la_m68k_gnu_pltenter (Elf32_Sym *sym __attribute__ ((unused)),
		      unsigned int ndx __attribute__ ((unused)),
		      uintptr_t *refcook, uintptr_t *defcook,
		      La_m68k_regs *regs, unsigned int *flags,
		      const char *symname, long int *framesizep)
{
  unsigned long int *sp = (unsigned long int *) regs->lr_sp;

  print_enter (refcook, defcook, symname, sp[1], sp[2], sp[3], *flags);

  /* No need to copy anything, we will not need the parameters in any case.  */
  *framesizep = 0;

  return sym->st_value;
}

unsigned int
la_m68k_gnu_pltexit (Elf32_Sym *sym, unsigned int ndx, uintptr_t *refcook,
		     uintptr_t *defcook,
		     const struct La_m68k_regs *inregs,
		     struct La_m68k_retval *outregs,
		     const char *symname)
{
  print_exit (refcook, defcook, symname, outregs->lrv_d0);

  return 0;
}
