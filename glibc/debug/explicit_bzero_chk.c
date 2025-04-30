/* Generic implementation of __explicit_bzero_chk.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Torbjorn Granlund (tege@sics.se).

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

/* This is the generic definition of __explicit_bzero_chk.  The
   __explicit_bzero_chk symbol is used as the implementation of
   explicit_bzero throughout glibc.  If this file is overriden by an
   architecture, both __explicit_bzero_chk and
   __explicit_bzero_chk_internal have to be defined (the latter not as
   an IFUNC).  */

#include <string.h>

void
__explicit_bzero_chk (void *dst, size_t len, size_t dstlen)
{
  /* Inline __memset_chk to avoid a PLT reference to __memset_chk.  */
  if (__glibc_unlikely (dstlen < len))
    __chk_fail ();
  memset (dst, '\0', len);
  /* Compiler barrier.  */
  asm volatile ("" ::: "memory");
}

/* libc-internal references use the hidden
   __explicit_bzero_chk_internal symbol.  This is necessary if
   __explicit_bzero_chk is implemented as an IFUNC because some
   targets do not support hidden references to IFUNC symbols.  */
strong_alias (__explicit_bzero_chk, __explicit_bzero_chk_internal)
