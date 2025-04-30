/* Make sure dlopen/dlclose are not marked as leaf functions.
   See bug-dl-leaf-lib.c for details.

   Copyright (C) 2013-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Mike Frysinger <vapier@gentoo.org>

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

extern void check_val_init (void);
extern void check_val_fini (void);

__attribute__ ((__constructor__))
void construct (void)
{
  check_val_init ();
}

__attribute__ ((__destructor__))
void destruct (void)
{
  check_val_fini ();
}
