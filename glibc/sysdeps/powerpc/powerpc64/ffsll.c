/* Find first set bit in a word, counted from least significant end.
   For PowerPC.
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

#define ffsl __something_else
#include <limits.h>
#include <string.h>

#undef	ffs

int
__ffsll (long long int x)
{
  int cnt;

  asm ("cntlzd %0,%1" : "=r" (cnt) : "r" (x & -x));
  return 64 - cnt;
}
weak_alias (__ffsll, ffsll)
#undef ffsl
weak_alias (__ffsll, ffsl)
