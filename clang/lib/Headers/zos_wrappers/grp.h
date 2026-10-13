/*===----------------------------- grp.h ----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_GRP_H
#define __ZOS_WRAPPERS_GRP_H
#if __has_include_next(<grp.h>)
#include_next <grp.h>
#ifdef __grp
#undef __grp
#define __grp __grp
#endif
#endif /* __has_include_next(<grp.h>) */
#endif /* __ZOS_WRAPPERS_GRP_H */
