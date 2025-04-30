/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef _NORM2_H_
#define _NORM2_H_

#include "stdioInterf.h"
#include "fioMacros.h"

#if defined(__AVX2__) && defined(DESC_I8)
#define NORM2_REAL4 norm2_avx2_real4_i8_
#define NORM2_REAL8 norm2_avx2_real8_i8_
#elif defined(__AVX2__) // implies !defined(DESC_I8)
#define NORM2_REAL4 norm2_avx2_real4_
#define NORM2_REAL8 norm2_avx2_real8_
#elif defined(DESC_I8) // implies !defined(__AVX2__)
#define NORM2_REAL4 norm2_real4_i8_
#define NORM2_REAL8 norm2_real8_i8_
#else // implies !defined(__AVX2__) && !defined(DESC_I8)
#define NORM2_REAL4 norm2_real4_
#define NORM2_REAL8 norm2_real8_
#endif

#endif
