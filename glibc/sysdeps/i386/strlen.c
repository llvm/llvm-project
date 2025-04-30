/* Determine the length of a string.  For Intel 80x86, x>=3.
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

#include <string.h>

size_t
strlen (const char *str)
{
  int cnt;

  asm("cld\n"			/* Search forward.  */
      /* Some old versions of gas need `repne' instead of `repnz'.  */
      "repnz\n"			/* Look for a zero byte.  */
      "scasb" /* %0, %1, %3 */ :
      "=c" (cnt) : "D" (str), "0" (-1), "a" (0));

  return -2 - cnt;
}
libc_hidden_builtin_def (strlen)
