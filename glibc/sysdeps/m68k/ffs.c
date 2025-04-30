/* ffs -- find first set bit in a word, counted from least significant end.
   For mc68020, mc68030, mc68040.
   This file is part of the GNU C Library.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#define ffsl __something_else
#include <string.h>

#undef	ffs

#if	defined (__GNUC__) && defined (__mc68020__)

int
__ffs (int x)
{
  int cnt;

  asm ("bfffo %1{#0:#0},%0" : "=d" (cnt) : "dm" (x & -x));

  return 32 - cnt;
}
weak_alias (__ffs, ffs)
libc_hidden_def (__ffs)
libc_hidden_builtin_def (ffs)
#undef ffsl
weak_alias (__ffs, ffsl)

#else

#include <string/ffs.c>

#endif
