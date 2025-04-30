/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#define versionsort __no_versionsort_decl
#include <dirent.h>
#undef versionsort
#include <string.h>

int
__versionsort64 (const struct dirent64 **a, const struct dirent64 **b)
{
  return __strverscmp ((*a)->d_name, (*b)->d_name);
}

#if _DIRENT_MATCHES_DIRENT64
weak_alias (__versionsort64, versionsort64)
weak_alias (__versionsort64, versionsort)
#else
# include <shlib-compat.h>
versioned_symbol (libc, __versionsort64, versionsort64, GLIBC_2_2);
# if SHLIB_COMPAT(libc, GLIBC_2_1, GLIBC_2_2)
#  include <olddirent.h>

int
attribute_compat_text_section
__old_versionsort64 (const struct __old_dirent64 **a,
		     const struct __old_dirent64 **b)
{
  return __strverscmp ((*a)->d_name, (*b)->d_name);
}

compat_symbol (libc, __old_versionsort64, versionsort64, GLIBC_2_1);
# endif /* SHLIB_COMPAT(libc, GLIBC_2_1, GLIBC_2_2)  */
#endif /* _DIRENT_MATCHES_DIRENT64  */
