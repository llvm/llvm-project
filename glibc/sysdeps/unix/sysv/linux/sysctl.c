/* sysctl function stub.
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

#include <errno.h>
#include <stddef.h>
#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_32)
int attribute_compat_text_section
___sysctl (int *name, int nlen, void *oldval, size_t *oldlenp,
           void *newval, size_t newlen)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, ___sysctl, sysctl, GLIBC_2_0);

# if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_17)
strong_alias (___sysctl, ___sysctl2)
compat_symbol (libc, ___sysctl2, __sysctl, GLIBC_2_2);
# endif
#endif
