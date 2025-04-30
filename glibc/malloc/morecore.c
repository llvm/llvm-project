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

#if defined(SHARED) || defined(USE_MTAG)
static bool __always_fail_morecore = false;
#endif

/* Allocate INCREMENT more bytes of data space,
   and return the start of data space, or NULL on errors.
   If INCREMENT is negative, shrink data space.  */
void *
__glibc_morecore (ptrdiff_t increment)
{
#if defined(SHARED) || defined(USE_MTAG)
  if (__always_fail_morecore)
    return NULL;
#endif

  void *result = (void *) __sbrk (increment);
  if (result == (void *) -1)
    return NULL;

  return result;
}
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libc, __glibc_morecore, __default_morecore, GLIBC_2_0);
#endif
