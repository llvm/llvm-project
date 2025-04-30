/* Dummy audit library for test-audit-threads.

   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <elf.h>
#include <link.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

/* We must use a dummy LD_AUDIT module to force the dynamic loader to
   *not* update the real PLT, and instead use a cached value for the
   lazy resolution result.  It is the update of that cached value that
   we are testing for correctness by doing this.  */

/* Library to be audited.  */
#define LIB "tst-audit-threads-mod2.so"
/* CALLNUM is the number of retNum functions.  */
#define CALLNUM 7999

#define CONCATX(a, b) __CONCAT (a, b)

static int previous = 0;

unsigned int
la_version (unsigned int ver)
{
  return 1;
}

unsigned int
la_objopen (struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
  return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

uintptr_t
CONCATX(la_symbind, __ELF_NATIVE_CLASS) (ElfW(Sym) *sym,
					unsigned int ndx,
					uintptr_t *refcook,
					uintptr_t *defcook,
					unsigned int *flags,
					const char *symname)
{
  const char * retnum = "retNum";
  char * num = strstr (symname, retnum);
  int n;
  /* Validate if the symbols are getting called in the correct order.
     This code is here to verify binutils does not optimize out the PLT
     entries that require the symbol binding.  */
  if (num != NULL)
    {
      n = atoi (num);
      assert (n >= previous);
      assert (n <= CALLNUM);
      previous = n;
    }
  return sym->st_value;
}
