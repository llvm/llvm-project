/*===----------------------------- math.h ----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_MATH_H
#define __ZOS_WRAPPERS_MATH_H
#if __has_include_next(<math.h>)
#include_next <math.h>
#ifdef __math
#undef __math
#define __math __math
#endif
#ifndef __BFP__
#ifdef __cplusplus
extern "C"
#endif
    double fabs(double x) __THROW;
#endif
#if !defined(__LP64__) && !defined(__BFP__)
#ifdef __C99
#pragma map(tgammaf, "\174\174TGMFH9")
#pragma map(tgamma, "\174\174TGMAH9")
#endif
#endif
#endif /* __has_include_next(<math.h>) */
#endif /* __ZOS_WRAPPERS_MATH_H */
