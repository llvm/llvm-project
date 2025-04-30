/* Test case for i386 preserved registers in dynamic linker.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <link.h>
#include <bits/wordsize.h>
#include <gnu/lib-names.h>

unsigned int
la_version (unsigned int v)
{
  setlinebuf (stdout);

  printf ("version: %u\n", v);

  char buf[20];
  sprintf (buf, "%u", v);

  return v;
}

void
la_activity (uintptr_t *cookie, unsigned int flag)
{
  const char *flagstr;
  switch (flag)
    {
    case LA_ACT_CONSISTENT:
      flagstr = "consistent";
      break;
    case LA_ACT_ADD:
      flagstr = "add";
      break;
    case LA_ACT_DELETE:
      flagstr = "delete";
      break;
    default:
      printf ("activity: unknown activity %u\n", flag);
      return;
    }
  printf ("activity: %s\n", flagstr);
}

char *
la_objsearch (const char *name, uintptr_t *cookie, unsigned int flag)
{
  const char *flagstr;
  switch (flag)
    {
    case LA_SER_ORIG:
      flagstr = "LA_SET_ORIG";
      break;
    case LA_SER_LIBPATH:
      flagstr = "LA_SER_LIBPATH";
      break;
    case LA_SER_RUNPATH:
      flagstr = "LA_SER_RUNPATH";
      break;
    case LA_SER_CONFIG:
      flagstr = "LA_SER_CONFIG";
      break;
    case LA_SER_DEFAULT:
      flagstr = "LA_SER_DEFAULT";
      break;
    case LA_SER_SECURE:
      flagstr = "LA_SER_SECURE";
      break;
    default:
      printf ("objsearch: %s, unknown flag %d\n", name, flag);
      return (char *) name;
    }

  printf ("objsearch: %s, %s\n", name, flagstr);
  return (char *) name;
}

unsigned int
la_objopen (struct link_map *l, Lmid_t lmid, uintptr_t *cookie)
{
  printf ("objopen: %ld, %s\n", lmid, l->l_name);

  return 3;
}

void
la_preinit (uintptr_t *cookie)
{
  printf ("preinit\n");
}

unsigned int
la_objclose  (uintptr_t *cookie)
{
  printf ("objclose\n");
  return 0;
}

uintptr_t
la_symbind32 (Elf32_Sym *sym, unsigned int ndx, uintptr_t *refcook,
	      uintptr_t *defcook, unsigned int *flags, const char *symname)
{
  printf ("symbind32: symname=%s, st_value=%#lx, ndx=%u, flags=%u\n",
	  symname, (long int) sym->st_value, ndx, *flags);

  return sym->st_value;
}

#include "tst-audit.h"

ElfW(Addr)
pltenter (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
	  uintptr_t *defcook, La_regs *regs, unsigned int *flags,
	  const char *symname, long int *framesizep)
{
  printf ("pltenter: symname=%s, st_value=%#lx, ndx=%u, flags=%u\n",
	  symname, (long int) sym->st_value, ndx, *flags);

  if (strcmp (symname, "audit1_test") == 0
      || strcmp (symname, "audit2_test") == 0)
    {
      if (regs->lr_eax != 1
	  || regs->lr_edx != 2
	  || regs->lr_ecx != 3)
	abort ();

      *framesizep = 200;
    }

  return sym->st_value;
}

unsigned int
pltexit (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
	 uintptr_t *defcook, const La_regs *inregs, La_retval *outregs,
	 const char *symname)
{
  printf ("pltexit: symname=%s, st_value=%#lx, ndx=%u, retval=%tu\n",
	  symname, (long int) sym->st_value, ndx,
	  (ptrdiff_t) outregs->int_retval);

  if (strcmp (symname, "audit1_test") == 0
      || strcmp (symname, "audit2_test") == 0)
    {
      if (inregs->lr_eax != 1
	  || inregs->lr_edx != 2
	  || inregs->lr_ecx != 3)
	abort ();

      if (strcmp (symname, "audit1_test") == 0)
	{
	  long long x = ((unsigned long long) outregs->lrv_eax
			 | (unsigned long long) outregs->lrv_edx << 32);

	  if (x != 30)
	    abort ();
	}
      else if (strcmp (symname, "audit2_test") == 0)
	{
	  if (outregs->lrv_st0 != 30)
	    abort ();
	}
    }

  return 0;
}
