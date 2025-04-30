/* Implement simple hashing table with string based keys.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Ulrich Drepper <drepper@gnu.ai.mit.edu>, October 1994.

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

#ifndef hashval_t
# define hashval_t unsigned long int
#endif
#include <limits.h>		/* For CHAR_BIT.  */

hashval_t
compute_hashval (const void *key, size_t keylen)
{
  size_t cnt;
  hashval_t hval;

  /* Compute the hash value for the given string.  The algorithm
     is taken from [Aho,Sethi,Ullman], modified to reduce the number of
     collisions for short strings with very varied bit patterns.
     See http://www.clisp.org/haible/hashfunc.html.  */
  cnt = 0;
  hval = keylen;
  while (cnt < keylen)
    {
      hval = (hval << 9) | (hval >> (sizeof hval * CHAR_BIT - 9));
      hval += (hashval_t) ((const unsigned char *) key)[cnt++];
    }
  return hval != 0 ? hval : ~((hashval_t) 0);
}
