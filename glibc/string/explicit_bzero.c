/* Erasure of sensitive data, generic implementation.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* An assembler implementation of explicit_bzero can be created as an
   assembler alias of an optimized bzero implementation.
   Architecture-specific implementations also need to define
   __explicit_bzero_chk.  */

#include <string.h>

/* glibc-internal users use __explicit_bzero_chk, and explicit_bzero
   redirects to that.  */
#undef explicit_bzero

/* Set LEN bytes of S to 0.  The compiler will not delete a call to
   this function, even if S is dead after the call.  */
void
explicit_bzero (void *s, size_t len)
{
  memset (s, '\0', len);
  /* Compiler barrier.  */
  asm volatile ("" ::: "memory");
}
