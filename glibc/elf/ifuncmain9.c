/* Test for IFUNC handling with local definitions.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* This test is based on gcc.dg/attr-ifunc-4.c.  */

#include <config.h>

# include <stdbool.h>
# include <stdio.h>

/* Do not use the test framework, so that the process setup is not
   disturbed.  */

static volatile int implementation_called;
static volatile int resolver_called;

/* Just a random constant, to check that we called the right
   function.  */
enum { random_constant = 0x3a88d66d };

static int
implementation (void)
{
  ++implementation_called;
  return random_constant;
}

static __typeof__ (implementation) *
inhibit_stack_protector
resolver (void)
{
  ++resolver_called;
  return implementation;
}

static int magic (void) __attribute__ ((ifunc ("resolver")));

int
main (void)
{
  bool errors = false;

  if (implementation_called != 0)
    {
      printf ("error: initial value of implementation_called is not zero:"
              " %d\n", implementation_called);
      errors = true;
    }

  /* This can be zero if the reference is bound lazily.  */
  printf ("info: initial value of resolver_called: %d\n", resolver_called);

  int magic_value = magic ();
  if (magic_value != random_constant)
    {
      printf ("error: invalid magic value: 0x%x\n", magic_value);
      errors = true;
    }

  printf ("info: resolver_called value: %d\n", resolver_called);
  if (resolver_called == 0)
    {
      /* In theory, the resolver could be called multiple times if
         several relocations are needed.  */
      puts ("error: invalid resolver_called value (must not be zero)");
      errors = true;
    }

  printf ("info: implementation_called value: %d\n", implementation_called);
  if (implementation_called != 1)
    {
      puts ("error: invalid implementation_called value (must be 1)");
      errors = true;
    }

  return errors;
}
