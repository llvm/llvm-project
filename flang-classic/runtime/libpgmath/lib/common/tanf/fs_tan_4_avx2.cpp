
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#ifndef __TAN_F_AVX2_128_H__
#define __TAN_F_AVX2_128_H__

#include <immintrin.h>
#include "common_tanf.h"
#define CONFIG 1
#include "helperavx2_128.h"
#include "tan_f_vec.h"

extern "C" vfloat __attribute__ ((noinline)) __fs_tan_4_avx2(vfloat const a);

vfloat __attribute__ ((noinline))
__fs_tan_4_avx2(vfloat const a)
{
	return __tan_f_vec(a);
}

#endif // __TAN_F_AVX2_128_H__

