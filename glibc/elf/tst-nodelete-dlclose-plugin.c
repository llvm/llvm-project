/* Bug 11941: Improper assert map->l_init_called in dlclose.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* This DSO simulates a plugin with a dependency on the
   primary DSO loaded by the appliation.  */
#include <stdio.h>

extern void primary_reference (void);

void
plugin_func (void)
{
  printf ("INFO: Calling plugin function.\n");
  /* Need a reference to the DSO to ensure that a potential --as-needed
     doesn't remove the DT_NEEDED entry which we rely upon to ensure
     destruction ordering.  */
  primary_reference ();
}

__attribute__ ((destructor))
static void
plugin_dtor (void)
{
  printf ("INFO: Calling plugin destructor.\n");
}
