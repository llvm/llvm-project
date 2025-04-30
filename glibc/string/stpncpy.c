/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef _LIBC
# include <string.h>
#else
# include <sys/types.h>
#endif

#ifndef STPNCPY
# ifdef weak_alias
#  define STPNCPY	__stpncpy
weak_alias (__stpncpy, stpncpy)
# else
#  define STPNCPY	stpncpy
# endif
#endif

/* Copy no more than N characters of SRC to DEST, returning the address of
   the terminating '\0' in DEST, if any, or else DEST + N.  */
char *
STPNCPY (char *dest, const char *src, size_t n)
{
  size_t size = __strnlen (src, n);
  memcpy (dest, src, size);
  dest += size;
  if (size == n)
    return dest;
  return memset (dest, '\0', n - size);
}
#ifdef weak_alias
libc_hidden_def (__stpncpy)
#endif
