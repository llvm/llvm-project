
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#ifndef __TAN_F_AVX512_H__
#define __TAN_F_AVX512_H__

#include <assert.h>
#include <immintrin.h>
#define CONFIG 1
#include "helperavx512f.h"
#ifndef TARGET_OSX_X8664
#include "common_tanf.h"
#include "tan_f_vec.h"
#endif

extern "C" vfloat __attribute__ ((noinline)) __fs_tan_16_avx512(vfloat const a);

vfloat __attribute__ ((noinline))
__fs_tan_16_avx512(vfloat const a)
{
#ifndef TARGET_OSX_X8664
	return __tan_f_vec(a);
#else
        assert(0);
        return ((vfloat) _mm512_set1_epi32(0));
#endif
}

#endif // __TAN_F_AVX512_H__

