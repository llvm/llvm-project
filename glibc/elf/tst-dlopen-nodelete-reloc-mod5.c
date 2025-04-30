/* Test propagation of NODELETE to an already-loaded object via relocation.
   NODELETE helper module.
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

#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

/* Defined in tst-dlopen-nodelete-reloc-mod3.so.  The dependency is
   expressed via DT_NEEDED on the intermediate DSO
   tst-dlopen-nodelete-reloc-mod3.so.  */
extern bool may_finalize_mod3;

static void __attribute__ ((destructor))
fini (void)
{
  if (!may_finalize_mod3)
    {
      puts ("error: tst-dlopen-nodelete-reloc-mod5.so destructor"
            " called too early");
      _exit (1);
    }
}
