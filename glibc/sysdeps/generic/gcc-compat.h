/* Macros for checking required GCC compatibility.  Generic version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

/* This is the base file.  More-specific sysdeps/.../gcc-compat.h files
   can define GCC_COMPAT_VERSION and then #include_next this file.  */

#ifndef _GENERIC_GCC_COMPAT_H
#define _GENERIC_GCC_COMPAT_H 1

/* This is the macro that gets used in #if tests in code: true iff
   the library we build must be compatible with user code built by
   GCC version MAJOR.MINOR.  */
#define GCC_COMPAT(major, minor)        \
  (GCC_COMPAT_VERSION <= GCC_VERSION (major, minor))

/* This is how we compose an integer from major and minor version
   numbers, for comparison.  */
#define GCC_VERSION(major, minor)       \
  (((major) << 16) + (minor))

#ifndef GCC_COMPAT_VERSION
/* GCC 2.7.2 was current at the time of the glibc-2.0 release.
   We assume nothing before that ever mattered.  */
# define GCC_COMPAT_VERSION     GCC_VERSION (2, 7)
#endif

#endif
