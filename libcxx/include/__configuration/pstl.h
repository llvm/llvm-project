// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_PSTL_H
#define _LIBCPP___CONFIGURATION_PSTL_H

#include <__config_site>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#define _PSTL_PRAGMA(x) _Pragma(#x)

// Enable SIMD for compilers that support OpenMP 4.0
#if (defined(_OPENMP) && _OPENMP >= 201307)

#  define _PSTL_UDR_PRESENT
#  define _PSTL_PRAGMA_SIMD _PSTL_PRAGMA(omp simd)
#  define _PSTL_PRAGMA_DECLARE_SIMD _PSTL_PRAGMA(omp declare simd)
#  define _PSTL_PRAGMA_SIMD_REDUCTION(PRM) _PSTL_PRAGMA(omp simd reduction(PRM))
#  define _PSTL_PRAGMA_SIMD_SCAN(PRM) _PSTL_PRAGMA(omp simd reduction(inscan, PRM))
#  define _PSTL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM) _PSTL_PRAGMA(omp scan inclusive(PRM))
#  define _PSTL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM) _PSTL_PRAGMA(omp scan exclusive(PRM))

// Declaration of reduction functor, where
// NAME - the name of the functor
// OP - type of the callable object with the reduction operation
// omp_in - refers to the local partial result
// omp_out - refers to the final value of the combiner operator
// omp_priv - refers to the private copy of the initial value
// omp_orig - refers to the original variable to be reduced
#  define _PSTL_PRAGMA_DECLARE_REDUCTION(NAME, OP)                                                                     \
    _PSTL_PRAGMA(omp declare reduction(NAME:OP : omp_out(omp_in)) initializer(omp_priv = omp_orig))

#elif defined(_LIBCPP_COMPILER_CLANG_BASED)

#  define _PSTL_PRAGMA_SIMD _Pragma("clang loop vectorize(enable) interleave(enable)")
#  define _PSTL_PRAGMA_DECLARE_SIMD
#  define _PSTL_PRAGMA_SIMD_REDUCTION(PRM) _Pragma("clang loop vectorize(enable) interleave(enable)")
#  define _PSTL_PRAGMA_SIMD_SCAN(PRM) _Pragma("clang loop vectorize(enable) interleave(enable)")
#  define _PSTL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM)
#  define _PSTL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM)
#  define _PSTL_PRAGMA_DECLARE_REDUCTION(NAME, OP)

#else // (defined(_OPENMP) && _OPENMP >= 201307)

#  define _PSTL_PRAGMA_SIMD
#  define _PSTL_PRAGMA_DECLARE_SIMD
#  define _PSTL_PRAGMA_SIMD_REDUCTION(PRM)
#  define _PSTL_PRAGMA_SIMD_SCAN(PRM)
#  define _PSTL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM)
#  define _PSTL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM)
#  define _PSTL_PRAGMA_DECLARE_REDUCTION(NAME, OP)

#endif // (defined(_OPENMP) && _OPENMP >= 201307)

#define _PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED

#endif // _LIBCPP___CONFIGURATION_PSTL_H
