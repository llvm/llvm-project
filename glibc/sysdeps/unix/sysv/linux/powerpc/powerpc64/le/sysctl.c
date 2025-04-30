/* sysctl function stub.  powerpc64le version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* powerpc64le is special because it has an ABI baseline of 2.17, but
   still includes the __sysctl symbol.  */

#ifdef SHARED

# include <sysdeps/unix/sysv/linux/sysctl.c>

strong_alias (___sysctl, ___sysctl2)
compat_symbol (libc, ___sysctl2, __sysctl, GLIBC_2_2);

#endif
