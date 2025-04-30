/* PowerPC specific sotruss-lib functions.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.

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

#ifdef __powerpc64__
# if _CALL_ELF != 2
#  define LA_PPC_REGS          La_ppc64_regs
#  define LA_PPC_RETVAL        La_ppc64_retval
#  define LA_PPC_GNU_PLTENTER  la_ppc64_gnu_pltenter
#  define LA_PPC_GNU_PLTEXIT   la_ppc64_gnu_pltexit
# else
#  define LA_PPC_REGS          La_ppc64v2_regs
#  define LA_PPC_RETVAL        La_ppc64v2_retval
#  define LA_PPC_GNU_PLTENTER  la_ppc64v2_gnu_pltenter
#  define LA_PPC_GNU_PLTEXIT   la_ppc64v2_gnu_pltexit
# endif
# else
# define LA_PPC_REGS           La_ppc32_regs
# define LA_PPC_RETVAL         La_ppc32_retval
# define LA_PPC_GNU_PLTENTER   la_ppc32_gnu_pltenter
# define LA_PPC_GNU_PLTEXIT    la_ppc32_gnu_pltexit
#endif

ElfW(Addr)
LA_PPC_GNU_PLTENTER (ElfW(Sym) *sym __attribute__ ((unused)),
		     unsigned int ndx __attribute__ ((unused)),
		     uintptr_t *refcook, uintptr_t *defcook,
		     LA_PPC_REGS *regs, unsigned int *flags,
		     const char *symname, long int *framesizep)
{
  print_enter (refcook, defcook, symname,
	       regs->lr_reg[0], regs->lr_reg[1], regs->lr_reg[2], *flags);

  /* No need to copy anything, we will not need the parameters in any case.  */
  *framesizep = 0;

  return sym->st_value;
}

unsigned int
LA_PPC_GNU_PLTEXIT (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
		    uintptr_t *defcook,
		    const struct LA_PPC_REGS *inregs,
		    struct LA_PPC_RETVAL *outregs, const char *symname)
{
  print_exit (refcook, defcook, symname, outregs->lrv_r3);

  return 0;
}
