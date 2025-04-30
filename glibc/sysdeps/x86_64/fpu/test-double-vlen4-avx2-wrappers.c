/* Wrapper part of tests for AVX2 ISA versions of vector math functions.
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

#include "test-double-vlen4.h"
#include "test-math-vector-sincos.h"
#include <immintrin.h>

#undef VEC_SUFF
#define VEC_SUFF _vlen4_avx2

#define VEC_TYPE __m256d

VECTOR_WRAPPER (WRAPPER_NAME (cos), _ZGVdN4v_cos)
VECTOR_WRAPPER (WRAPPER_NAME (sin), _ZGVdN4v_sin)
VECTOR_WRAPPER (WRAPPER_NAME (log), _ZGVdN4v_log)
VECTOR_WRAPPER (WRAPPER_NAME (exp), _ZGVdN4v_exp)
VECTOR_WRAPPER_ff (WRAPPER_NAME (pow), _ZGVdN4vv_pow)

#ifndef __ILP32__
# define VEC_INT_TYPE __m256i
#else
# define VEC_INT_TYPE __m128i
#endif

VECTOR_WRAPPER_fFF_2 (WRAPPER_NAME (sincos), _ZGVdN4vvv_sincos)
