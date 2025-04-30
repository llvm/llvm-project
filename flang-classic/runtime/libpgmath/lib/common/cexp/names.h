
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if !(defined __NAMES_H_INCLUDED__)
#define __NAMES_H_INCLUDED__ 1

#if (defined __AVX512F__)
    #define sizeof_simd 64
#elif (defined __AVX2__)
    #define sizeof_simd 32
#else //SSE
    #define sizeof_simd 16
#endif

#if !(defined _MEASURE_BASELINE_)
    #define _VECTOR_FUNCTION_
    // double precision scalar portable code is faster than
    // scalarized SIMD
    #define __USE_PORTABLE_CODE
#else
    #undef _VL
#endif

#if !(defined _VL)
    #if (defined _VECTOR_FUNCTION_)
        //#define _VL (sizeof_simd / sizeof(float _Complex))
        #if (defined __AVX512F__)
            #define _VL 4
        #elif (defined __AVX2__)
            #define _VL 2
        #else //SSE
            #define _VL 1
        #endif
    #else
        #define _VL 1
    #endif
#endif

#if ((_VL) == (1))
    // scalar code
    #define __SLEEF_AVX2_128__
#elif ((_VL) == (2))
    #define __SLEEF_AVX2_256__
#elif ((_VL) == (4))
    #define __SLEEF_AVX512__
#endif

#if ( sizeof_simd < (_VL*8) )
#error: SIMD size sizeof_simd is less than vector length _VL * sizeof(float _Complex)
#endif

#if !(defined _CPU)
#error: please define _CPU - specific suffix to a function name
#endif

#if (defined _VECTOR_FUNCTION_)
    #if ((_VL) > (1))
        #define cexp_vec         JOIN4(__fz_exp_,_VL,_,_CPU)
    #else
        #undef cexp_vec
    #endif
#endif

#define cexp_scalar_default_abi  JOIN4(__fz_exp_,1,_,_CPU)

#if (defined _SCALAR_WITH_VECTOR_ABI_)
    #define cexp_scalar_vector_abi JOIN4(__fz_exp_,1v,_,_CPU)
    #define cexp_scalar cexp_scalar_vector_abi
#else
    #define cexp_scalar cexp_scalar_default_abi
#endif

#if !(defined F_VISIBILITY)
    #define F_VISIBILITY extern
    //#define F_VISIBILITY static __attribute((noinline))
    //#define F_VISIBILITY static __attribute((always_inline)) inline
#endif

#if ((_VL) == (1))
    #if (defined _SCALAR_WITH_VECTOR_ABI_)
        #define F_VISIBILITY_SCALAR static INLINE
        #define F_VISIBILITY_SCALAR_VECTOR F_VISIBILITY
    #else
        #define F_VISIBILITY_SCALAR F_VISIBILITY
    #endif
    #define F_VISIBILITY_VEC static INLINE
#else
    #define F_VISIBILITY_VEC F_VISIBILITY
#endif

#endif //__NAMES_H_INCLUDED__
