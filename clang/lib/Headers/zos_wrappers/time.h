/*===----------------------------- time.h ----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_TIME_H
#define __ZOS_WRAPPERS_TIME_H
#if __has_include_next(<time.h>)
#include_next <time.h>
#ifdef __time
#undef __time
#define __time __time
#endif
#endif /* __has_include_next(<time.h>) */
#endif /* __ZOS_WRAPPERS_TIME_H */
