/* Old-versioned functions for binary compatibility with glibc-2.0.
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

#include <hurd.h>

/* This file provides definitions for binary compatibility with
   the GLIBC_2.0 version set for the libc.so.0.2 soname.

   These definitions can be removed when the soname changes.  */

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

void
attribute_compat_text_section
_hurd_proc_init_compat_20 (char **argv)
{
  _hurd_proc_init (argv, NULL, 0);
}
compat_symbol (libc, _hurd_proc_init_compat_20, _hurd_proc_init, GLIBC_2_0);

#endif
