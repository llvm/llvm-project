/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
# include "config.h"
#endif

#include <stdio.h>
#include <sys/types.h>

#if defined _LIBC || defined STDC_HEADERS
# include <stdlib.h>
# include <string.h>
#else
char *malloc ();
#endif

#undef __strndup
#undef strndup

#ifndef weak_alias
# define __strndup strndup
#endif

char *
__strndup (const char *s, size_t n)
{
  size_t len = __strnlen (s, n);
  char *new = (char *) malloc (len + 1);

  if (new == NULL)
    return NULL;

  new[len] = '\0';
  return (char *) memcpy (new, s, len);
}
#ifdef libc_hidden_def
libc_hidden_def (__strndup)
#endif
#ifdef weak_alias
weak_alias (__strndup, strndup)
#endif
