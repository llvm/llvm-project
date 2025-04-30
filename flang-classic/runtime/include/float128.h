/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef _FLOAT128_H_
#define _FLOAT128_H_

#if defined(TARGET_LINUX_POWER) || defined(TARGET_LINUX_X8664)
typedef __float128 float128_t;
#else
/* __float128 is not available on AArch64 or other generic targets;
 * on AArch64 at least, long double is 128 bits in size.
 *
 * FIXME: On Windows long double size is 64 bits, same as double is.
 * As workaround we used clang's '-mlong-double-128' compiler option
 * to force long double to be 128 bits.
 * With newer Windows SDK it causes an assertion in one of system header.
 * Until the final solution we should use float128_t type with this limitation
 * on Windows.
 * Please see the issue: https://github.com/flang-compiler/flang/issues/1334
 */
typedef long double float128_t;
#endif

#endif /* _FLOAT128_H_ */
