/*===-------------------------- variant.h ----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_VARIANT_H
#define __ZOS_WRAPPERS_VARIANT_H
#if __has_include_next(<variant.h>)
#include_next <variant.h>
#ifdef __variant
#undef __variant
#define __variant __variant
#endif
#endif /* __has_include_next(<variant.h>) */
#endif /* __ZOS_WRAPPERS_VARIANT_H */
