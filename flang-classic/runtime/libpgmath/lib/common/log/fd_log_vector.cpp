
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#if !(defined _CPU)
#error: please define _CPU - specific suffix to a function name
#endif

#if !(defined _VL)
#error: please define _VL - Number of elements per vector register
#endif


#include <immintrin.h>
#define CONFIG 1
#if ((_VL) == (2))
#include "helperavx2_128.h"
#elif ((_VL) == (4))
#include "helperavx2.h"
#elif ((_VL) == (8))
#include "helperavx512f.h"
#endif


#define _JOIN4(a,b,c,d) a##b##c##d
#define JOIN4(a,b,c,d) _JOIN4(a,b,c,d)

#define log_d_vec JOIN4(__fd_log_,_VL,_,_CPU)

extern "C" vdouble log_d_vec(vdouble const);

#include <log_d_vec.h>
