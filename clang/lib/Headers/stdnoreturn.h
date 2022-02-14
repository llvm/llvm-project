/*===---- stdnoreturn.h - Standard header for noreturn macro ---------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __STDNORETURN_H
#define __STDNORETURN_H

#define noreturn _Noreturn
#define __noreturn_is_defined 1

#if __STDC_VERSION__ > 201710L &&                                              \
    !defined(_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS)
/* The noreturn macro is deprecated in C2x. */
#pragma clang deprecated(noreturn)

/* Including the header file in C2x is also deprecated. */
#warning "the '<stdnoreturn.h>' header is deprecated in C2x"
#endif

#endif /* __STDNORETURN_H */
