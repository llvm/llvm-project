/*===---- omp_libmextras.h -----host functions not defined in libm         -===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

// NVIDIA and AMD define device math functions that are not in libm.
// They do this for CUDA and HIP respectively.   For OpenMP, we need a
// fallback function for host execution. These functions are defined here.
// c and c++ users must include these with #include <omp_libmextras.h>

#ifndef __OMP_LIBMEXTRAS_H__
#define __OMP_LIBMEXTRAS_H__

#ifndef _OPENMP
#error "This file is for OpenMP compilation only."
#endif

// Host definitions of functions not in libm.
#if !defined(__NVPTX__) && !defined(__AMDGCN__)
float sinpif(const float x) { return (sinf(x * M_PI)); }
double sinpi(const double x) { return (sin(x * M_PI)); }
float cospif(const float x) { return (cosf(x * M_PI)); }
double cospi(const double x) { return (cos(x * M_PI)); }
#endif

#endif // __OMP_LIBMEXTRAS_H__
