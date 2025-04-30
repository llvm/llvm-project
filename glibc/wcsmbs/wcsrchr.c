/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper, <drepper@gnu.ai.mit.edu>

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

#ifndef WCSRCHR
# define WCSRCHR wcsrchr
#endif

/* Find the last occurrence of WC in WCS.  */
wchar_t *
WCSRCHR (const wchar_t *wcs, const wchar_t wc)
{
  wchar_t *retval = NULL;

#define ITERATION(index)		\
  ({					\
    if (*wcs == wc)			\
      retval = (wchar_t*) wcs;		\
    *wcs++ != L'\0';	\
  })

#ifndef UNROLL_NTIMES
# define UNROLL_NTIMES 1
#endif

  while (1)
    UNROLL_REPEAT (UNROLL_NTIMES, ITERATION);

  return retval;
}
