/*===---- __stddef_null.h - Definition of NULL -----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#undef NULL
#ifdef __cplusplus
#if !defined(__MINGW32__) && !defined(_MSC_VER)
#define NULL __null
#else
#define NULL 0
#endif
#else
// Don't add any whitespaces in ((void*)0) below!
// musl (https://www.musl-libc.org/) redefines `NULL` as such and redefinition
// with a different expression, even in terms of a single whitespace, causes a
// warning.
#define NULL ((void*)0)
#endif
