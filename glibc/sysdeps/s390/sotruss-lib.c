/* Override generic sotruss-lib.c to define actual functions for s390.
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

#if __ELF_NATIVE_CLASS == 32
# define la_s390_gnu_pltenter	la_s390_32_gnu_pltenter
# define la_s390_gnu_pltexit	la_s390_32_gnu_pltexit
# define La_s390_regs		La_s390_32_regs
# define La_s390_retval		La_s390_32_retval
#else
# define la_s390_gnu_pltenter	la_s390_64_gnu_pltenter
# define la_s390_gnu_pltexit	la_s390_64_gnu_pltexit
# define La_s390_regs		La_s390_64_regs
# define La_s390_retval		La_s390_64_retval
#endif

ElfW(Addr)
la_s390_gnu_pltenter (ElfW(Sym) *sym,
		      unsigned int ndx __attribute__ ((unused)),
		      uintptr_t *refcook, uintptr_t *defcook,
		      La_s390_regs *regs, unsigned int *flags,
		      const char *symname, long int *framesizep)
{
  print_enter (refcook, defcook, symname,
	       regs->lr_r2, regs->lr_r3, regs->lr_r4, *flags);

  /* No need to copy anything, we will not need the parameters in any case.  */
  *framesizep = 0;

  return sym->st_value;
}

unsigned int
la_s390_gnu_pltexit (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
		     uintptr_t *defcook,
		     const struct La_s390_regs *inregs,
		     struct La_s390_retval *outregs, const char *symname)
{
  print_exit (refcook, defcook, symname, outregs->lrv_r2);

  return 0;
}
