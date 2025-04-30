/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, August 1995.

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

#include <stdlib.h>

#define TABLE_BASE 0x2e
#define TABLE_SIZE 0x4d

#define XX ((char)0x40)


static const char a64l_table[TABLE_SIZE] =
{
  /* 0x2e */                                                           0,  1,
  /* 0x30 */   2,  3,  4,  5,  6,  7,  8,  9, 10, 11, XX, XX, XX, XX, XX, XX,
  /* 0x40 */  XX, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
  /* 0x50 */  27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, XX, XX, XX, XX, XX,
  /* 0x60 */  XX, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
  /* 0x70 */  53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
};


long int
a64l (const char *string)
{
  const char *ptr = string;
  unsigned long int result = 0ul;
  const char *end = ptr + 6;
  int shift = 0;

  do
    {
      unsigned index;
      unsigned value;

      index = *ptr - TABLE_BASE;
      if ((unsigned int) index >= TABLE_SIZE)
	break;
      value = (int) a64l_table[index];
      if (value == (int) XX)
	break;
      ++ptr;
      result |= value << shift;
      shift += 6;
    }
  while (ptr != end);

  return (long int) result;
}
