/*===-- flang/Common/float80.h --------------------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/

/* This header is usable in both C and C++ code.
 * Isolates build compiler checks to determine if the 80-bit
 * floating point format is supported via a particular C type.
 * It defines CFloat80Type and CppFloat80Type aliases for this
 * C type.
 */

#ifndef FORTRAN_COMMON_FLOAT80_H_
#define FORTRAN_COMMON_FLOAT80_H_

#include "api-attrs.h"
#include <float.h>

#if LDBL_MANT_DIG == 64
#undef HAS_FLOAT80
#define HAS_FLOAT80 1
#endif

#if defined(RT_DEVICE_COMPILATION) && defined(__CUDACC__)
/*
 * 'long double' is treated as 'double' in the CUDA device code,
 * and there is no support for 80-bit floating point format.
 * This is probably true for most offload devices, so RT_DEVICE_COMPILATION
 * check should be enough. For the time being, guard it with __CUDACC__
 * as well.
 */
#undef HAS_FLOAT80
#endif

#if HAS_FLOAT80
typedef long double CFloat80Type;
typedef long double CppFloat80Type;
#endif

#endif /* FORTRAN_COMMON_FLOAT80_H_ */
