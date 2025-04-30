/* Test x86-specific floating-point environment (bug 16068): SSE part.
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

#include <cpuid.h>
#include <fenv.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static bool
have_sse2 (void)
{
  unsigned int eax, ebx, ecx, edx;

  if (!__get_cpuid (1, &eax, &ebx, &ecx, &edx))
    return false;

  return (edx & bit_SSE2) != 0;
}

static uint32_t
get_sse_mxcsr (void)
{
  uint32_t temp;
  __asm__ __volatile__ ("stmxcsr %0" : "=m" (temp));
  return temp;
}

static void
set_sse_mxcsr (uint32_t val)
{
  __asm__ __volatile__ ("ldmxcsr %0" : : "m" (val));
}

static void
set_sse_mxcsr_bits (uint32_t mask, uint32_t bits)
{
  uint32_t mxcsr = get_sse_mxcsr ();
  mxcsr = (mxcsr & ~mask) | bits;
  set_sse_mxcsr (mxcsr);
}

static int
test_sse_mxcsr_bits (const char *test, uint32_t mask, uint32_t bits)
{
  uint32_t mxcsr = get_sse_mxcsr ();
  printf ("Testing %s: mxcsr = %x\n", test, mxcsr);
  if ((mxcsr & mask) == bits)
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

#define MXCSR_FZ 0x8000
#define MXCSR_DAZ 0x40
#define MXCSR_DE 0x2
#define MXCSR_DM 0x100

static __attribute__ ((noinline)) int
sse_tests (void)
{
  int result = 0;
  fenv_t env1, env2;
  /* Test FZ bit.  */
  fegetenv (&env1);
  set_sse_mxcsr_bits (MXCSR_FZ, MXCSR_FZ);
  fegetenv (&env2);
  fesetenv (&env1);
  result |= test_sse_mxcsr_bits ("fesetenv FZ restoration",
				 MXCSR_FZ, 0);
  set_sse_mxcsr_bits (MXCSR_FZ, 0);
  fesetenv (&env2);
  result |= test_sse_mxcsr_bits ("fesetenv FZ restoration 2",
				 MXCSR_FZ, MXCSR_FZ);
  set_sse_mxcsr_bits (MXCSR_FZ, MXCSR_FZ);
  fesetenv (FE_NOMASK_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_NOMASK_ENV) FZ restoration",
				 MXCSR_FZ, 0);
  set_sse_mxcsr_bits (MXCSR_FZ, MXCSR_FZ);
  fesetenv (FE_DFL_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_DFL_ENV) FZ restoration",
				 MXCSR_FZ, 0);
  /* Test DAZ bit.  */
  set_sse_mxcsr_bits (MXCSR_DAZ, MXCSR_DAZ);
  fegetenv (&env2);
  fesetenv (&env1);
  result |= test_sse_mxcsr_bits ("fesetenv DAZ restoration",
				 MXCSR_DAZ, 0);
  set_sse_mxcsr_bits (MXCSR_DAZ, 0);
  fesetenv (&env2);
  result |= test_sse_mxcsr_bits ("fesetenv DAZ restoration 2",
				 MXCSR_DAZ, MXCSR_DAZ);
  set_sse_mxcsr_bits (MXCSR_DAZ, MXCSR_DAZ);
  fesetenv (FE_NOMASK_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_NOMASK_ENV) DAZ restoration",
				 MXCSR_DAZ, 0);
  set_sse_mxcsr_bits (MXCSR_DAZ, MXCSR_DAZ);
  fesetenv (FE_DFL_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_DFL_ENV) DAZ restoration",
				 MXCSR_DAZ, 0);
  /* Test DM bit.  */
  set_sse_mxcsr_bits (MXCSR_DM, 0);
  fegetenv (&env2);
  fesetenv (&env1);
  result |= test_sse_mxcsr_bits ("fesetenv DM restoration",
				 MXCSR_DM, MXCSR_DM);
  set_sse_mxcsr_bits (MXCSR_DM, MXCSR_DM);
  fesetenv (&env2);
  result |= test_sse_mxcsr_bits ("fesetenv DM restoration 2",
				 MXCSR_DM, 0);
  set_sse_mxcsr_bits (MXCSR_DM, 0);
  /* Presume FE_NOMASK_ENV should leave the "denormal operand"
     exception masked, as not a standard exception.  */
  fesetenv (FE_NOMASK_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_NOMASK_ENV) DM restoration",
				 MXCSR_DM, MXCSR_DM);
  set_sse_mxcsr_bits (MXCSR_DM, 0);
  fesetenv (FE_DFL_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_DFL_ENV) DM restoration",
				 MXCSR_DM, MXCSR_DM);
  /* Test DE bit.  */
  set_sse_mxcsr_bits (MXCSR_DE, MXCSR_DE);
  fegetenv (&env2);
  fesetenv (&env1);
  result |= test_sse_mxcsr_bits ("fesetenv DE restoration",
				 MXCSR_DE, 0);
  set_sse_mxcsr_bits (MXCSR_DE, 0);
  fesetenv (&env2);
  result |= test_sse_mxcsr_bits ("fesetenv DE restoration 2",
				 MXCSR_DE, MXCSR_DE);
  set_sse_mxcsr_bits (MXCSR_DE, MXCSR_DE);
  fesetenv (FE_NOMASK_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_NOMASK_ENV) DE restoration",
				 MXCSR_DE, 0);
  set_sse_mxcsr_bits (MXCSR_DE, MXCSR_DE);
  fesetenv (FE_DFL_ENV);
  result |= test_sse_mxcsr_bits ("fesetenv (FE_DFL_ENV) DE restoration",
				 MXCSR_DE, 0);
  return result;
}

static int
do_test (void)
{
  if (!have_sse2 ())
    {
      puts ("CPU does not support SSE2, cannot test");
      return 0;
    }
  return sse_tests ();
}

#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>
