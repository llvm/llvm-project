/*===----------------------------- locale.h
 *----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-------------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_LOCALE_H
#define __ZOS_WRAPPERS_LOCALE_H
#if defined(__MVS__)
#include_next <locale.h>
#ifdef __locale
#undef __locale
#define __locale __locale
#endif
#endif /* defined(__MVS__) */
#endif /* __ZOS_WRAPPERS_LOCALE_H */
