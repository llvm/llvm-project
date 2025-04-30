/* IFUNC resolver with CPU_FEATURE_ACTIVE.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <sys/platform/x86.h>

enum isa
{
  none,
  sse2,
  sse4_2,
  avx,
  avx2,
  avx512f
};

enum isa
__attribute__ ((__optimize__ ("-fno-stack-protector")))
get_isa (void)
{
  if (CPU_FEATURE_ACTIVE (AVX512F))
    return avx512f;
  if (CPU_FEATURE_ACTIVE (AVX2))
    return avx2;
  if (CPU_FEATURE_ACTIVE (AVX))
    return avx;
  if (CPU_FEATURE_ACTIVE (SSE4_2))
    return sse4_2;
  if (CPU_FEATURE_ACTIVE (SSE2))
    return sse2;
  return none;
}

static int
isa_sse2 (void)
{
  return sse2;
}

static int
isa_sse4_2 (void)
{
  return sse4_2;
}

static int
isa_avx (void)
{
  return avx;
}

static int
isa_avx2 (void)
{
  return avx2;
}

static int
isa_avx512f (void)
{
  return avx512f;
}

static int
isa_none (void)
{
  return none;
}

int foo (void) __attribute__ ((ifunc ("foo_ifunc")));

void *
__attribute__ ((__optimize__ ("-fno-stack-protector")))
foo_ifunc (void)
{
  switch (get_isa ())
    {
    case avx512f:
      return isa_avx512f;
    case avx2:
      return isa_avx2;
    case avx:
      return isa_avx;
    case sse4_2:
      return isa_sse4_2;
    case sse2:
      return isa_sse2;
    default:
      break;
    }
  return isa_none;
}
