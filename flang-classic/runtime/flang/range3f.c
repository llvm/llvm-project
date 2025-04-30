/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * Implements LIB3F range functions.
 */

#include "ent3f.h"

/* can remove !defined(OSF86) when float.h is corrected */
#if !defined(I386) && !defined(OSF86) && !defined(OSX86)
#include <float.h>
#else
#define FLT_EPSILON 1.19209290E-07F
#define FLT_MIN 1.17549435E-38F
#define FLT_MAX 3.40282347E+38F

#define DBL_EPSILON 2.2204460492503131E-16
#define DBL_MIN 2.2250738585072014E-308
#define DBL_MAX 1.7976931348623157E+308

#endif

float ENT3F(FLMIN, flmin)(void) { return FLT_MIN; }

float ENT3F(FLMAX, flmax)(void) { return FLT_MAX; }

float ENT3F(FFRAC, ffrac)(void) { return FLT_EPSILON; }

double ENT3F(DFLMIN, dflmin)(void) { return DBL_MIN; }

double ENT3F(DFLMAX, dflmax)(void) { return DBL_MAX; }

double ENT3F(DFFRAC, dffrac)(void) { return DBL_EPSILON; }

#ifdef TARGET_SUPPORTS_QUADFP
long double ENT3F(QFFRAC, qffrac)(void) { return LDBL_EPSILON; }
#endif

int ENT3F(INMAX, inmax)(void) { return 0x7fffffff; }
