/* wait4 -- wait for process to change state.  Linux/Alpha version.
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
   <http://www.gnu.org/licenses/>.  */

#include <shlib-compat.h>

#undef weak_alias
#define weak_alias(a, b)
#include <sysdeps/unix/sysv/linux/wait4.c>
#undef weak_alias
#define weak_alias(name, aliasname) _weak_alias (name, aliasname)
versioned_symbol (libc, __wait4, wait4, GLIBC_2_1);

/* GLIBC_2_0 version is implemented at osf_wait4.c.  */
