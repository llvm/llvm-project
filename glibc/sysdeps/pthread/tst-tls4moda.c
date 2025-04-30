/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <stddef.h>
#include <stdlib.h>

static __thread unsigned char foo [32]
  __attribute__ ((tls_model ("initial-exec"), aligned (sizeof (void *))));

void
test1 (void)
{
  size_t s;

  for (s = 0; s < sizeof (foo); ++s)
    {
      if (foo [s])
	abort ();
      foo [s] = s;
    }
}

void
test2 (void)
{
  size_t s;

  for (s = 0; s < sizeof (foo); ++s)
    {
      if (foo [s] != s)
	abort ();
      foo [s] = sizeof (foo) - s;
    }
}
