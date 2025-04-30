/* Test case for preserved AVX registers in dynamic linker.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

int tst_audit4_aux (void);

static int
avx_enabled (void)
{
  unsigned int eax, ebx, ecx, edx;

  if (__get_cpuid (1, &eax, &ebx, &ecx, &edx) == 0
      || (ecx & (bit_AVX | bit_OSXSAVE)) != (bit_AVX | bit_OSXSAVE))
    return 0;

  /* Check the OS has AVX and SSE saving enabled.  */
  asm ("xgetbv" : "=a" (eax), "=d" (edx) : "c" (0));

  return (eax & 6) == 6;
}

static int
do_test (void)
{
  /* Run AVX test only if AVX is supported.  */
  if (avx_enabled ())
    return tst_audit4_aux ();
  else
    return 77;
}

#define TEST_FUNCTION do_test ()
#include "../../test-skeleton.c"
