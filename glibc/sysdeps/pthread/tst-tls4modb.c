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

static int i;
int bar;

static __thread void *foo [32 / sizeof (void *)]
  __attribute__ ((tls_model ("initial-exec"), aligned (sizeof (void *))))
  = { &i, &bar };

void
test1 (void)
{
  size_t s;

  if (foo [0] != &i || foo [1] != &bar)
    abort ();

  foo [0] = NULL;
  foo [1] = NULL;
  for (s = 0; s < sizeof (foo) / sizeof (void *); ++s)
    {
      if (foo [s])
	abort ();
      foo [s] = &foo[s];
    }
}

void
test2 (void)
{
  size_t s;

  for (s = 0; s < sizeof (foo) / sizeof (void *); ++s)
    {
      if (foo [s] != &foo [s])
	abort ();
      foo [s] = &foo [s ^ 1];
    }
}
