/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

/* The object of this file should be installed as libmcheck.a,
   so one can do -lmcheck to turn on mcheck.  */

#include <malloc.h>
#include <mcheck.h>
#include <shlib-compat.h>

static void
turn_on_mcheck (void)
{
  mcheck (NULL);
}

void (*__malloc_initialize_hook) (void) = turn_on_mcheck;
compat_symbol_reference (libc, __malloc_initialize_hook,
                         __malloc_initialize_hook, GLIBC_2_0);
