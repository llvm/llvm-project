/*===--------------------------- string.h ----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_STRING_H
#define __ZOS_WRAPPERS_STRING_H
#if __has_include_next(<string.h>)
#include_next <string.h>
#ifdef __string
#undef __string
#define __string __string
#endif
#endif /* __has_include_next(<string.h>) */
#endif /* __ZOS_WRAPPERS_STRING_H */
