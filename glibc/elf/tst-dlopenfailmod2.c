/* Module which depends on on a NODELETE module, and can be loaded.
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

#include <stdio.h>

/* Force linking against tst-dlopenfailnodelmod.so.  */
void no_delete_mod_function (void);
void *function_reference = no_delete_mod_function;

static void __attribute__ ((constructor))
init (void)
{
  puts ("info: tst-dlopenfailmod2.so constructor invoked");
}
