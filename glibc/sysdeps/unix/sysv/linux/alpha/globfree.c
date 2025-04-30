/* Compat globfree.  Linux/alpha version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#define globfree64 __no_globfree64_decl
#include <sys/types.h>
#include <glob.h>
#include <shlib-compat.h>

#define globfree(pglob) \
  __new_globfree (pglob)

extern void __new_globfree (glob_t *__pglob);

#include <posix/globfree.c>

#undef globfree64

versioned_symbol (libc, __new_globfree, globfree, GLIBC_2_1);
libc_hidden_ver (__new_globfree, globfree)

weak_alias (__new_globfree, globfree64)
libc_hidden_ver (__new_globfree, globfree64)
