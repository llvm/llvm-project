/* First module for NODELETE test defining a unique symbol (with DT_NEEDED).
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

#include "tst-dlopen-nodelete-reloc.h"

#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

/* Just a flag here, not used for NODELETE processing.  */
bool may_finalize_mod12 = false;

/* Explicit instantiation.  This produces a unique symbol definition
   which is not referenced by the library itself, so the library is
   not marked NODELETE.  */
template struct unique_symbol<12>;

static void __attribute__ ((destructor))
fini (void)
{
  if (!may_finalize_mod12)
    {
      puts ("error: tst-dlopen-nodelete-reloc-mod12.so destructor"
            " called too early");
      _exit (1);
    }
}
