/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1995.

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

#include <wchar.h>
#include <loop_unroll.h>


#ifdef WCSCPY
# define __wcscpy WCSCPY
#endif

/* Copy SRC to DEST.  */
wchar_t *
__wcscpy (wchar_t *dest, const wchar_t *src)
{
#ifndef UNROLL_NTIMES
  return __wmemcpy (dest, src, __wcslen (src) + 1);
#else
  /* Some architectures might have costly tail function call (powerpc
     for instance) where wmemcpy call overhead for smalls sizes might
     be more costly than just unrolling the main loop.  */
  wchar_t *wcp = dest;

#define ITERATION(index)		\
  ({					\
     wchar_t c = *src++;		\
     *wcp++ = c;			\
     c != L'\0';			\
  })

  while (1)
    UNROLL_REPEAT(UNROLL_NTIMES, ITERATION);
  return dest;
#endif
}
#ifndef WCSCPY
weak_alias (__wcscpy, wcscpy)
libc_hidden_def (__wcscpy)
#endif
