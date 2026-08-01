/*===----------------------------- stdlib.h --------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_STDLIB_H
#define __ZOS_WRAPPERS_STDLIB_H
#if __has_include_next(<stdlib.h>)
#include_next <stdlib.h>
#ifdef _EXT
#ifndef __CS1
#undef __cs
#endif
#endif /* _EXT */
#endif /* __has_include_next(<stdlib.h>) */
#endif /* __ZOS_WRAPPERS_STDLIB_H */
