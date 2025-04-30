/* Test STT_GNU_IFUNC resolver with second argument.
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

#include <stdint.h>
#include <sys/auxv.h>
#include <sys/ifunc.h>
#include <support/check.h>

static int
one (void)
{
  return 1;
}

static uint64_t saved_arg1;
static __ifunc_arg_t saved_arg2;

/* extern visible ifunc symbol.  */
int
foo (void);

void *
foo_ifunc (uint64_t, const __ifunc_arg_t *) __asm__ ("foo");
__asm__(".type foo, %gnu_indirect_function");

void *
inhibit_stack_protector
foo_ifunc (uint64_t arg1, const __ifunc_arg_t *arg2)
{
  saved_arg1 = arg1;
  if (arg1 & _IFUNC_ARG_HWCAP)
      saved_arg2 = *arg2;
  return (void *) one;
}

static int
do_test (void)
{
  TEST_VERIFY (foo () == 1);
  TEST_VERIFY (saved_arg1 & _IFUNC_ARG_HWCAP);
  TEST_COMPARE ((uint32_t)saved_arg1, (uint32_t)getauxval (AT_HWCAP));
  TEST_COMPARE (saved_arg2._size, sizeof (__ifunc_arg_t));
  TEST_COMPARE (saved_arg2._hwcap, getauxval (AT_HWCAP));
  TEST_COMPARE (saved_arg2._hwcap2, getauxval (AT_HWCAP2));
  return 0;
}

#include <support/test-driver.c>
