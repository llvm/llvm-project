/* Copy memory to memory until the specified number of bytes
   has been copied with error checking.  Overlap is handled correctly.
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

#include <string.h>
#include <memcopy.h>

#ifndef MEMMOVE_CHK
# define MEMMOVE_CHK __memmove_chk
#endif

void *
MEMMOVE_CHK (void *dest, const void *src, size_t len, size_t destlen)
{
  if (__glibc_unlikely (destlen < len))
    __chk_fail ();

  return memmove (dest, src, len);
}
