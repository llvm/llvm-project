/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1997.

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

#include <nss.h>

/* This is from libc/db/hash/hash_func.c, hash3 is static there */
/*
 * This is INCREDIBLY ugly, but fast.  We break the string up into 8 byte
 * units.  On the first time through the loop we get the "leftover bytes"
 * (strlen % 8).  On every other iteration, we perform 8 HASHC's so we handle
 * all 8 bytes.  Essentially, this saves us 7 cmp & branch instructions.  If
 * this routine is heavily used enough, it's worth the ugly coding.
 *
 * OZ's original sdbm hash
 */
uint32_t
__nss_hash (const void *keyarg, size_t len)
{
  const unsigned char *key;
  size_t loop;
  uint32_t h;

#define HASHC   h = *key++ + 65599 * h

  h = 0;
  key = keyarg;
  if (len > 0)
    {
      loop = (len + 8 - 1) >> 3;
      switch (len & (8 - 1))
        {
        case 0:
          do
            {
              HASHC;
              /* FALLTHROUGH */
            case 7:
              HASHC;
              /* FALLTHROUGH */
            case 6:
              HASHC;
              /* FALLTHROUGH */
            case 5:
              HASHC;
              /* FALLTHROUGH */
            case 4:
              HASHC;
              /* FALLTHROUGH */
            case 3:
              HASHC;
              /* FALLTHROUGH */
            case 2:
              HASHC;
              /* FALLTHROUGH */
            case 1:
              HASHC;
            }
	  while (--loop);
        }
    }
  return h;
}

libc_hidden_def (__nss_hash)
