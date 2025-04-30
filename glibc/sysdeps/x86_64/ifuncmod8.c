/* Test IFUNC selector with floating-point parameters.
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

#include <emmintrin.h>

void * foo_ifunc (void) __asm__ ("foo");
__asm__(".type foo, %gnu_indirect_function");

static float
foo_impl (float x)
{
  return x + 1;
}

void *
inhibit_stack_protector
foo_ifunc (void)
{
  __m128i xmm = _mm_set1_epi32 (-1);
  asm volatile ("movdqa %0, %%xmm0" : : "x" (xmm) : "xmm0" );
  return foo_impl;
}
