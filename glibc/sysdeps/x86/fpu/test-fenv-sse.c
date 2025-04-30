/* Test floating-point environment includes SSE state (bug 16064).
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
#include <stdio.h>

static bool
have_sse2 (void)
{
  unsigned int eax, ebx, ecx, edx;

  if (!__get_cpuid (1, &eax, &ebx, &ecx, &edx))
    return false;

  return (edx & bit_SSE2) != 0;
}

static __attribute__ ((noinline)) int
sse_tests (void)
{
  int ret = 0;
  fenv_t base_env;
  if (fegetenv (&base_env) != 0)
    {
      puts ("fegetenv (&base_env) failed");
      return 1;
    }
  if (fesetround (FE_UPWARD) != 0)
    {
      puts ("fesetround (FE_UPWARD) failed");
      return 1;
    }
  if (fesetenv (&base_env) != 0)
    {
      puts ("fesetenv (&base_env) failed");
      return 1;
    }
  volatile float a = 1.0f, b = FLT_MIN, c;
  c = a + b;
  if (c != 1.0f)
    {
      puts ("fesetenv did not restore rounding mode");
      ret = 1;
    }
  if (fesetround (FE_DOWNWARD) != 0)
    {
      puts ("fesetround (FE_DOWNWARD) failed");
      return 1;
    }
  if (feupdateenv (&base_env) != 0)
    {
      puts ("feupdateenv (&base_env) failed");
      return 1;
    }
  volatile float d = -FLT_MIN, e;
  e = a + d;
  if (e != 1.0f)
    {
      puts ("feupdateenv did not restore rounding mode");
      ret = 1;
    }
  if (fesetround (FE_UPWARD) != 0)
    {
      puts ("fesetround (FE_UPWARD) failed");
      return 1;
    }
  fenv_t upward_env;
  if (feholdexcept (&upward_env) != 0)
    {
      puts ("feholdexcept (&upward_env) failed");
      return 1;
    }
  if (fesetround (FE_DOWNWARD) != 0)
    {
      puts ("fesetround (FE_DOWNWARD) failed");
      return 1;
    }
  if (fesetenv (&upward_env) != 0)
    {
      puts ("fesetenv (&upward_env) failed");
      return 1;
    }
  e = a + d;
  if (e != 1.0f)
    {
      puts ("fesetenv did not restore rounding mode from feholdexcept");
      ret = 1;
    }
  if (fesetround (FE_UPWARD) != 0)
    {
      puts ("fesetround (FE_UPWARD) failed");
      return 1;
    }
  if (fesetenv (FE_DFL_ENV) != 0)
    {
      puts ("fesetenv (FE_DFL_ENV) failed");
      return 1;
    }
  c = a + b;
  if (c != 1.0f)
    {
      puts ("fesetenv (FE_DFL_ENV) did not restore rounding mode");
      ret = 1;
    }
  return ret;
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
