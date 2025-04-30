/* Test fesetenv (FE_DFL_ENV) and fesetenv (FE_NOMASK_ENV) clear
   exceptions (bug 19181).  SSE version.
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
#include <stdbool.h>

static bool
have_sse2 (void)
{
  unsigned int eax, ebx, ecx, edx;

  if (!__get_cpuid (1, &eax, &ebx, &ecx, &edx))
    return false;

  return (edx & bit_SSE2) != 0;
}

#define CHECK_CAN_TEST						\
  do								\
    {								\
      if (!have_sse2 ())					\
	{							\
	  puts ("CPU does not support SSE2, cannot test");	\
	  return 0;						\
	}							\
    }								\
  while (0)

#include <test-fenv-clear-main.c>
