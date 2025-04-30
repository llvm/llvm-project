/* Definitions of macros to access `dev_t' values.  Hurd version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _BITS_SYSMACROS_H
#define _BITS_SYSMACROS_H 1

#ifndef _SYS_SYSMACROS_H
# error "Never include <bits/sysmacros.h> directly; use <sys/sysmacros.h> instead."
#endif

/* The Hurd version of dev_t in glibc is a 32-bit quantity, with 8-bit
   major and 24-bit minor numbers.  The encoding is mmmmMMmm, where M is a
   hex digit of the major number and m is a hex digit of the minor number.  */

#define __SYSMACROS_DECLARE_MAJOR(DECL_TEMPL)                   \
  DECL_TEMPL(unsigned int, major, (__dev_t __dev))

#define __SYSMACROS_DEFINE_MAJOR(DECL_TEMPL)                    \
  __SYSMACROS_DECLARE_MAJOR (DECL_TEMPL)                        \
  {                                                             \
    return ((__dev & (__dev_t) 0x0000ff00u) >> 8);              \
  }

#define __SYSMACROS_DECLARE_MINOR(DECL_TEMPL)                   \
  DECL_TEMPL(unsigned int, minor, (__dev_t __dev))

#define __SYSMACROS_DEFINE_MINOR(DECL_TEMPL)                    \
  __SYSMACROS_DECLARE_MINOR (DECL_TEMPL)                        \
  {                                                             \
    return (__dev & (__dev_t) 0xffff00ff);                      \
  }

#define __SYSMACROS_DECLARE_MAKEDEV(DECL_TEMPL)                 \
  DECL_TEMPL(__dev_t, makedev, (unsigned int __major, unsigned int __minor))

#define __SYSMACROS_DEFINE_MAKEDEV(DECL_TEMPL)                  \
  __SYSMACROS_DECLARE_MAKEDEV (DECL_TEMPL)                      \
  {                                                             \
    __dev_t __dev;                                              \
    __dev  = (((__dev_t) (__major & 0x000000ffu)) << 8);        \
    __dev |= (((__dev_t) (__minor & 0xffff00ffu)) << 0);        \
    return __dev;                                               \
  }

#endif /* bits/sysmacros.h */
