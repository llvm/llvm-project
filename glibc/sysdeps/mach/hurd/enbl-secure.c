/* Define and initialize the `__libc_enable_secure' flag.  Hurd version.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

/* There is no need for this file in the Hurd; it is just a placeholder
   to prevent inclusion of the sysdeps/generic version.
   In the shared library, the `__libc_enable_secure' variable is defined
   by the dynamic linker in dl-sysdep.c and set there.
   In the static library, it is defined in init-first.c and set there.  */

#include <libc-internal.h>

void
__libc_init_secure (void)
{
}
