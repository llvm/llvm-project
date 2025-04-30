/* Check for invalid audit version (BZ#24122).
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <link.h>
#include <stdlib.h>

unsigned int
la_version (unsigned int version)
{
  /* The audit specification says that a version of 0 or a version
     greater than any version supported by the dynamic loader shall
     cause the module to be ignored.  */
  return 0;
}

void
la_activity (uintptr_t *cookie, unsigned int flag)
{
  exit (EXIT_FAILURE);
}

char *
la_objsearch (const char *name, uintptr_t *cookie, unsigned int flag)
{
  exit (EXIT_FAILURE);
}

unsigned int
la_objopen (struct link_map *map, Lmid_t lmid, uintptr_t * cookie)
{
  exit (EXIT_FAILURE);
}

void
la_preinit (uintptr_t * cookie)
{
  exit (EXIT_FAILURE);
}

uintptr_t
#if __ELF_NATIVE_CLASS == 32
la_symbind32 (Elf32_Sym *sym, unsigned int ndx, uintptr_t *refcook,
              uintptr_t *defcook, unsigned int *flags, const char *symname)
#else
la_symbind64 (Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
              uintptr_t *defcook, unsigned int *flags, const char *symname)
#endif
{
  exit (EXIT_FAILURE);
}

unsigned int
la_objclose (uintptr_t * cookie)
{
  exit (EXIT_FAILURE);
}

#include <tst-audit.h>
#if (!defined (pltenter) || !defined (pltexit) || !defined (La_regs) \
     || !defined (La_retval) || !defined (int_retval))
# error "architecture specific code needed in sysdeps/CPU/tst-audit.h"
#endif

ElfW(Addr)
pltenter (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
          uintptr_t *defcook, La_regs *regs, unsigned int *flags,
          const char *symname, long int *framesizep)
{
  exit (EXIT_FAILURE);
}

unsigned int
pltexit (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
         uintptr_t *defcook, const La_regs *inregs, La_retval *outregs,
         const char *symname)
{
  exit (EXIT_FAILURE);
}
