/* Test netinet/in.h endian-conversion macros always return the correct type.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <netinet/in.h>
#include <stdint.h>

int i;
uint16_t u16;
uint32_t u32;

int
do_test (void)
{
  /* This is a compilation test.  */
  extern __typeof (htons (i)) u16;
  extern __typeof (ntohs (i)) u16;
  extern __typeof (htonl (i)) u32;
  extern __typeof (ntohl (i)) u32;
  (void) u16;
  (void) u32;
  return 0;
}

#include <support/test-driver.c>
