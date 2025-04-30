/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

#if     defined(TARGET_X8664)
/*
 * For X8664, implement both SSE and AVX versions of __mth_i_floor using ISA
 * instruction extensions.
 *
 * Using inline assembly allows both the SSE and AVX versions of the routine
 * to be compiled in a single unit.
 *
 * The following asm statements is equivalent to:
 *      return _mm_cvtss_f32(_mm_floor_ss(_mm_set1_ps(x), _mm_set1_ps(x)));
 * But without the need for separate compiliations for SSE4.1 and AVX ISA
 * extensions.
 */

float
__mth_i_floor_sse(float x)
{
  __asm__(
    "roundss $0x1,%0,%0"
    :"+x"(x)
    );
  return x;
}

float
__mth_i_floor_avx(float x)
{
  __asm__(
    "vroundss $0x1,%0,%0,%0"
    :"+x"(x)
    );
  return x;
}
#endif

float
__mth_i_floor(float x)
{
  return floorf(x);
}
