/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Ulrich Drepper <drepper@cygnus.com>, 1995.

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

#ifndef STRING_TYPE
# define STRING_TYPE char
# define STRCOLL strcoll
# define STRCOLL_L __strcoll_l
# define USE_HIDDEN_DEF
#endif

#include "../locale/localeinfo.h"


int
STRCOLL (const STRING_TYPE *s1, const STRING_TYPE *s2)
{
  return STRCOLL_L (s1, s2, _NL_CURRENT_LOCALE);
}

#ifdef USE_HIDDEN_DEF
libc_hidden_def (STRCOLL)
#endif
