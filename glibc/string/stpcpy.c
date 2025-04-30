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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#define NO_MEMPCPY_STPCPY_REDIRECT
#include <string.h>

#undef __stpcpy
#undef stpcpy

#ifndef STPCPY
# define STPCPY __stpcpy
#endif

/* Copy SRC to DEST, returning the address of the terminating '\0' in DEST.  */
char *
STPCPY (char *dest, const char *src)
{
  size_t len = strlen (src);
  return memcpy (dest, src, len + 1) + len;
}
weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
