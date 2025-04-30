/* Test for vector sincos ABI.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <math-tests-arch.h>

extern int test_sincos_abi (void);

int arch_check = 1;

static void
check_arch (void)
{
  CHECK_ARCH_EXT;
  arch_check = 0;
}

static int
do_test (void)
{
  check_arch ();

  if (arch_check)
    return 77;

  return test_sincos_abi ();
}

#define TEST_FUNCTION do_test ()
#include "../../../test-skeleton.c"
