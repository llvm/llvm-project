/*===---- stdarg.h - Variable argument handling ----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(__STDARG_H) || defined(__need___va_list) ||                       \
    defined(__need_va_list) || defined(__need_va_arg) ||                       \
    defined(__need___va_copy) || defined(__need_va_copy)

#if !defined(__need___va_list) && !defined(__need_va_list) &&                  \
    !defined(__need_va_arg) && !defined(__need___va_copy) &&                   \
    !defined(__need_va_copy)
#define __STDARG_H
#define __need___va_list
#define __need_va_list
#define __need_va_arg
#define __need___va_copy
/* GCC always defines __va_copy, but does not define va_copy unless in c99 mode
 * or -ansi is not specified, since it was not part of C90.
 */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) ||              \
    (defined(__cplusplus) && __cplusplus >= 201103L) ||                        \
    !defined(__STRICT_ANSI__)
#define __need_va_copy
#endif
#endif

#ifdef __need___va_list
#ifndef __GNUC_VA_LIST
#define __GNUC_VA_LIST
typedef __builtin_va_list __gnuc_va_list;
#endif
#undef __need___va_list
#endif /* defined(__need___va_list) */

#ifdef __need_va_list
#ifndef _VA_LIST
typedef __builtin_va_list va_list;
#define _VA_LIST
#endif
#undef __need_va_list
#endif /* defined(__need_va_list) */

#ifdef __need_va_arg
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
/* C23 does not require the second parameter for va_start. */
#define va_start(ap, ...) __builtin_va_start(ap, 0)
#else
/* Versions before C23 do require the second parameter. */
#define va_start(ap, param) __builtin_va_start(ap, param)
#endif
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)
#undef __need_va_arg
#endif /* defined(__need_va_arg) */

#ifdef __need___va_copy
#define __va_copy(d,s) __builtin_va_copy(d,s)
#undef __need___va_copy
#endif /* defined(__need___va_copy) */

#ifdef __need_va_copy
#define va_copy(dest, src)  __builtin_va_copy(dest, src)
#undef __need_va_copy
#endif /* defined(__need_va_copy) */

#endif
