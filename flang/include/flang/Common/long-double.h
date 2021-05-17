/*===-- include/flang/Common/config.h -------------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

/* This header can be used by both C and C++. */

#ifndef FORTRAN_COMMON_LONG_DOUBLE_H
#define FORTRAN_COMMON_LONG_DOUBLE_H

#ifdef _MSC_VER /* no long double */
#undef LONG_DOUBLE
#elif __x86_64__ /* x87 extended precision */
#define LONG_DOUBLE 80
#else
#define LONG_DOUBLE 128
#endif

#endif /* FORTRAN_COMMON_LONG_DOUBLE_H */
