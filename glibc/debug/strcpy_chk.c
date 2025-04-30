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

#include <stddef.h>
#include <string.h>
#include <memcopy.h>

#undef strcpy

/* Copy SRC to DEST with checking of destination buffer overflow.  */
char *
__strcpy_chk (char *dest, const char *src, size_t destlen)
{
  size_t len = strlen (src);
  if (len >= destlen)
    __chk_fail ();

  return memcpy (dest, src, len + 1);
}
