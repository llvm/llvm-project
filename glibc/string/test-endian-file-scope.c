/* Test endian.h endian-conversion macros accepted at file scope.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <endian.h>
#include <stddef.h>

int i;

size_t s0 = sizeof (htobe16 (i));
size_t s1 = sizeof (htole16 (i));
size_t s2 = sizeof (be16toh (i));
size_t s3 = sizeof (le16toh (i));
size_t s4 = sizeof (htobe32 (i));
size_t s5 = sizeof (htole32 (i));
size_t s6 = sizeof (be32toh (i));
size_t s7 = sizeof (le32toh (i));
size_t s8 = sizeof (htobe64 (i));
size_t s9 = sizeof (htole64 (i));
size_t s10 = sizeof (be64toh (i));
size_t s11 = sizeof (le64toh (i));

static int
do_test (void)
{
  /* This is a compilation test.  */
  return 0;
}

#include <support/test-driver.c>
