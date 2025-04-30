/* ffs -- find first set bit in a word, counted from least significant end.
   S/390 version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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

#include <limits.h>
#define ffsl __something_else
#include <string.h>

#undef	ffs

/* ffs: find first bit set. This is defined the same way as
   the libc and compiler builtin ffs routines, therefore
   differs in spirit from the above ffz (man ffs).  */

int
__ffs (int x)
{
	int r;

	if (x == 0)
	  return 0;
	__asm__("    lr	  %%r1,%1\n"
		"    sr	  %0,%0\n"
		"    tml  %%r1,0xFFFF\n"
		"    jnz  0f\n"
		"    ahi  %0,16\n"
		"    srl  %%r1,16\n"
		"0:  tml  %%r1,0x00FF\n"
		"    jnz  1f\n"
		"    ahi  %0,8\n"
		"    srl  %%r1,8\n"
		"1:  tml  %%r1,0x000F\n"
		"    jnz  2f\n"
		"    ahi  %0,4\n"
		"    srl  %%r1,4\n"
		"2:  tml  %%r1,0x0003\n"
		"    jnz  3f\n"
		"    ahi  %0,2\n"
		"    srl  %%r1,2\n"
		"3:  tml  %%r1,0x0001\n"
		"    jnz  4f\n"
		"    ahi  %0,1\n"
		"4:"
		: "=&d" (r) : "d" (x) : "cc", "1" );
	return r+1;
}

weak_alias (__ffs, ffs)
libc_hidden_def (__ffs)
libc_hidden_builtin_def (ffs)
#if ULONG_MAX == UINT_MAX
#undef ffsl
weak_alias (__ffs, ffsl)
#endif
