/* Test x86-specific floating-point environment (bug 16068): x87 part.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <float.h>
#include <fpu_control.h>
#include <stdint.h>
#include <stdio.h>

static uint16_t
get_x87_cw (void)
{
  fpu_control_t cw;
  _FPU_GETCW (cw);
  return cw;
}

static void
set_x87_cw (uint16_t val)
{
  fpu_control_t cw = val;
  _FPU_SETCW (cw);
}

static void
set_x87_cw_bits (uint16_t mask, uint16_t bits)
{
  uint16_t cw = get_x87_cw ();
  cw = (cw & ~mask) | bits;
  set_x87_cw (cw);
}

static int
test_x87_cw_bits (const char *test, uint16_t mask, uint16_t bits)
{
  uint16_t cw = get_x87_cw ();
  printf ("Testing %s: cw = %x\n", test, cw);
  if ((cw & mask) == bits)
    {
      printf ("PASS: %s\n", test);
      return 0;
    }
  else
    {
      printf ("FAIL: %s\n", test);
      return 1;
    }
}

static uint16_t
get_x87_sw (void)
{
  uint16_t temp;
  __asm__ __volatile__ ("fnstsw %0" : "=a" (temp));
  return temp;
}

static void
set_x87_sw_bits (uint16_t mask, uint16_t bits)
{
  fenv_t temp;
  __asm__ __volatile__ ("fnstenv %0" : "=m" (temp));
  temp.__status_word = (temp.__status_word & ~mask) | bits;
  __asm__ __volatile__ ("fldenv %0" : : "m" (temp));
}

static int
test_x87_sw_bits (const char *test, uint16_t mask, uint16_t bits)
{
  uint16_t sw = get_x87_sw ();
  printf ("Testing %s: sw = %x\n", test, sw);
  if ((sw & mask) == bits)
    {
      printf ("PASS: %s\n", test);
      return 0;
    }
  else
    {
      printf ("FAIL: %s\n", test);
      return 1;
    }
}

#define X87_CW_PREC_MASK _FPU_EXTENDED

static int
do_test (void)
{
  int result = 0;
  fenv_t env1, env2;
  /* Test precision mask.  */
  fegetenv (&env1);
  set_x87_cw_bits (X87_CW_PREC_MASK, _FPU_SINGLE);
  fegetenv (&env2);
  fesetenv (&env1);
  result |= test_x87_cw_bits ("fesetenv precision restoration",
			      X87_CW_PREC_MASK, _FPU_EXTENDED);
  set_x87_cw_bits (X87_CW_PREC_MASK, _FPU_EXTENDED);
  fesetenv (&env2);
  result |= test_x87_cw_bits ("fesetenv precision restoration 2",
			      X87_CW_PREC_MASK, _FPU_SINGLE);
  set_x87_cw_bits (X87_CW_PREC_MASK, _FPU_DOUBLE);
  fesetenv (FE_NOMASK_ENV);
  result |= test_x87_cw_bits ("fesetenv (FE_NOMASK_ENV) precision restoration",
			      X87_CW_PREC_MASK, _FPU_EXTENDED);
  set_x87_cw_bits (X87_CW_PREC_MASK, _FPU_SINGLE);
  fesetenv (FE_DFL_ENV);
  result |= test_x87_cw_bits ("fesetenv (FE_DFL_ENV) precision restoration",
			      X87_CW_PREC_MASK, _FPU_EXTENDED);
  /* Test x87 denormal operand masking.  */
  set_x87_cw_bits (_FPU_MASK_DM, 0);
  fegetenv (&env2);
  fesetenv (&env1);
  result |= test_x87_cw_bits ("fesetenv denormal mask restoration",
			      _FPU_MASK_DM, _FPU_MASK_DM);
  set_x87_cw_bits (_FPU_MASK_DM, _FPU_MASK_DM);
  fesetenv (&env2);
  result |= test_x87_cw_bits ("fesetenv denormal mask restoration 2",
			      _FPU_MASK_DM, 0);
  set_x87_cw_bits (_FPU_MASK_DM, 0);
  /* Presume FE_NOMASK_ENV should leave the "denormal operand"
     exception masked, as not a standard exception.  */
  fesetenv (FE_NOMASK_ENV);
  result |= test_x87_cw_bits ("fesetenv (FE_NOMASK_ENV) denormal mask "
			      "restoration",
			      _FPU_MASK_DM, _FPU_MASK_DM);
  set_x87_cw_bits (_FPU_MASK_DM, 0);
  fesetenv (FE_DFL_ENV);
  result |= test_x87_cw_bits ("fesetenv (FE_DFL_ENV) denormal mask "
			      "restoration",
			      _FPU_MASK_DM, _FPU_MASK_DM);
  /* Test x87 denormal operand exception.  */
  set_x87_sw_bits (__FE_DENORM, __FE_DENORM);
  fegetenv (&env2);
  fesetenv (&env1);
  result |= test_x87_sw_bits ("fesetenv denormal exception restoration",
			      __FE_DENORM, 0);
  set_x87_sw_bits (__FE_DENORM, 0);
  fesetenv (&env2);
  result |= test_x87_sw_bits ("fesetenv denormal exception restoration 2",
			      __FE_DENORM, __FE_DENORM);
  set_x87_sw_bits (__FE_DENORM, __FE_DENORM);
  fesetenv (FE_NOMASK_ENV);
  result |= test_x87_sw_bits ("fesetenv (FE_NOMASK_ENV) exception restoration",
			      __FE_DENORM, 0);
  set_x87_sw_bits (__FE_DENORM, __FE_DENORM);
  fesetenv (FE_DFL_ENV);
  result |= test_x87_sw_bits ("fesetenv (FE_DFL_ENV) exception restoration",
			      __FE_DENORM, 0);
  return result;
}

#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>
