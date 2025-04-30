/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <string.h>

/* Copy memory like memcpy, but no return value required.  Can't alias
   to memcpy because it's not defined in the same translation
   unit.  */
void
__aeabi_memcpy (void *dest, const void *src, size_t n)
{
  memcpy (dest, src, n);
}

/* Versions of the above which may assume memory alignment.  */
strong_alias (__aeabi_memcpy, __aeabi_memcpy4)
strong_alias (__aeabi_memcpy, __aeabi_memcpy8)
