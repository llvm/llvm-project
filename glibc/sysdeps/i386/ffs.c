/* ffs -- find first set bit in a word, counted from least significant end.
   For Intel 80x86, x>=3.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#define ffsl __something_else
#include <string.h>

#undef	ffs

#ifdef	__GNUC__

int
__ffs (int x)
{
  int cnt;
  int tmp;

  asm ("xorl %0,%0\n"		/* Set CNT to zero.  */
       "bsfl %2,%1\n"		/* Count low bits in X and store in %1.  */
       "jz 1f\n"		/* Jump if OK, i.e. X was non-zero.  */
       "leal 1(%1),%0\n"	/* Return bsfl-result plus one on %0.  */
       "1:" : "=&a" (cnt), "=r" (tmp) : "rm" (x));

  return cnt;
}
weak_alias (__ffs, ffs)
libc_hidden_def (__ffs)
libc_hidden_builtin_def (ffs)
#undef ffsl
weak_alias (__ffs, ffsl)

#else
#include <string/ffs.c>
#endif
