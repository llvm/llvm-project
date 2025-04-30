/* Test hton/ntoh functions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <endian.h>
#include <stdio.h>
#include <sys/types.h>
#include <netinet/in.h>

#if BYTE_ORDER == BIG_ENDIAN
# define TEST(orig, swapped, fct) \
  if ((fct (orig)) != (orig)) {						      \
    printf ("Failed for %s -> %#x\n", #fct "(" #orig ")", fct (orig));	      \
    result = 1;								      \
  }
#elif BYTE_ORDER == LITTLE_ENDIAN
# define TEST(orig, swapped, fct) \
  if ((fct (orig)) != (swapped)) {					      \
    printf ("Failed for %s -> %#x\n", #fct "(" #orig ")", fct (orig));	      \
    result = 1;								      \
  }
#else
# error "Bah, what kind of system do you use?"
#endif

uint32_t lo = 0x67452301;
uint16_t foo = 0x1234;

int
main (void)
{
  int result = 0;

  TEST (0x67452301, 0x01234567, htonl);
  TEST (0x67452301, 0x01234567, (htonl));
  TEST (0x67452301, 0x01234567, ntohl);
  TEST (0x67452301, 0x01234567, (ntohl));

  TEST (lo, 0x01234567, htonl);
  TEST (lo, 0x01234567, (htonl));
  TEST (lo, 0x01234567, ntohl);
  TEST (lo, 0x01234567, (ntohl));

  TEST (0x1234, 0x3412, htons);
  TEST (0x1234, 0x3412, (htons));
  TEST (0x1234, 0x3412, ntohs);
  TEST (0x1234, 0x3412, (ntohs));

  TEST (foo, 0x3412, htons);
  TEST (foo, 0x3412, (htons));
  TEST (foo, 0x3412, ntohs);
  TEST (foo, 0x3412, (ntohs));

  return result;
}
