/*
 * UFC-crypt: ultra fast crypt(3) implementation
 *
 * Copyright (C) 1991-2021 Free Software Foundation, Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; see the file COPYING.LIB.  If not,
 * see <https://www.gnu.org/licenses/>.
 *
 * @(#)crypt.c	2.25 12/20/96
 *
 * Semiportable C version
 *
 */

#include "crypt-private.h"

#ifdef _UFC_32_

/*
 * 32 bit version
 */

#define SBA(sb, v) (*(long32*)((char*)(sb)+(v)))

void
_ufc_doit_r (ufc_long itr, struct crypt_data * __restrict __data,
	     ufc_long *res)
{
  int i;
  long32 s, *k;
  long32 *sb01 = (long32*)__data->sb0;
  long32 *sb23 = (long32*)__data->sb2;
  long32 l1, l2, r1, r2;

  l1 = (long32)res[0]; l2 = (long32)res[1];
  r1 = (long32)res[2]; r2 = (long32)res[3];

  while(itr--) {
    k = (long32*)__data->keysched;
    for(i=8; i--; ) {
      s = *k++ ^ r1;
      l1 ^= SBA(sb01, s & 0xffff); l2 ^= SBA(sb01, (s & 0xffff)+4);
      l1 ^= SBA(sb01, s >>= 16  ); l2 ^= SBA(sb01, (s         )+4);
      s = *k++ ^ r2;
      l1 ^= SBA(sb23, s & 0xffff); l2 ^= SBA(sb23, (s & 0xffff)+4);
      l1 ^= SBA(sb23, s >>= 16  ); l2 ^= SBA(sb23, (s         )+4);

      s = *k++ ^ l1;
      r1 ^= SBA(sb01, s & 0xffff); r2 ^= SBA(sb01, (s & 0xffff)+4);
      r1 ^= SBA(sb01, s >>= 16  ); r2 ^= SBA(sb01, (s         )+4);
      s = *k++ ^ l2;
      r1 ^= SBA(sb23, s & 0xffff); r2 ^= SBA(sb23, (s & 0xffff)+4);
      r1 ^= SBA(sb23, s >>= 16  ); r2 ^= SBA(sb23, (s         )+4);
    }
    s=l1; l1=r1; r1=s; s=l2; l2=r2; r2=s;
  }
  res[0] = l1; res[1] = l2; res[2] = r1; res[3] = r2;
}

#endif

#ifdef _UFC_64_

/*
 * 64 bit version
 */

#define SBA(sb, v) (*(long64*)((char*)(sb)+(v)))

void
_ufc_doit_r (ufc_long itr, struct crypt_data * __restrict __data,
	     ufc_long *res)
{
  int i;
  long64 l, r, s, *k;
  long64 *sb01 = (long64*)__data->sb0;
  long64 *sb23 = (long64*)__data->sb2;

  l = (((long64)res[0]) << 32) | ((long64)res[1]);
  r = (((long64)res[2]) << 32) | ((long64)res[3]);

  while(itr--) {
    k = (long64*)__data->keysched;
    for(i=8; i--; ) {
      s = *k++ ^ r;
      l ^= SBA(sb23, (s       ) & 0xffff);
      l ^= SBA(sb23, (s >>= 16) & 0xffff);
      l ^= SBA(sb01, (s >>= 16) & 0xffff);
      l ^= SBA(sb01, (s >>= 16)         );

      s = *k++ ^ l;
      r ^= SBA(sb23, (s       ) & 0xffff);
      r ^= SBA(sb23, (s >>= 16) & 0xffff);
      r ^= SBA(sb01, (s >>= 16) & 0xffff);
      r ^= SBA(sb01, (s >>= 16)         );
    }
    s=l; l=r; r=s;
  }

  res[0] = l >> 32; res[1] = l & 0xffffffff;
  res[2] = r >> 32; res[3] = r & 0xffffffff;
}

#endif
