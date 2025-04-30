/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

#if     defined(TARGET_X8664)
/*
 * For X8664, implement both SSE and AVX versions of __mth_i_ceil using ISA
 * instruction extensions.
 *
 * Using inline assembly allows both the SSE and AVX versions of the routine
 * to be compiled in a single unit.
 *
 * The following asm statements is equivalent to:
 *      return _mm_cvtss_f32(_mm_ceil_ss(_mm_set1_ps(x), _mm_set1_ps(x)));
 * But without the need for separate compiliations for SSE4.1 and AVX ISA
 * extensions.
 */

double
__mth_i_dceil_sse(double x)
{
  __asm__(
    "roundsd $0x2,%0,%0"
    :"+x"(x)
    );
  return x;
}

double
__mth_i_dceil_avx(double x)
{
  __asm__(
    "vroundsd $0x2,%0,%0,%0"
    :"+x"(x)
    );
  return x;
}
#endif

double
__mth_i_dceil(double x)
{
  return ceil(x);
}
