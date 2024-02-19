/*===---- __stddef_wchar.h - Definition of wchar_t -------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(__cplusplus) || (defined(_MSC_VER) && !_NATIVE_WCHAR_T_DEFINED)

#ifndef _WCHAR_T
#define _WCHAR_T

#ifdef _MSC_EXTENSIONS
#define _WCHAR_T_DEFINED
#endif

typedef __WCHAR_TYPE__ wchar_t;

#endif

#endif
