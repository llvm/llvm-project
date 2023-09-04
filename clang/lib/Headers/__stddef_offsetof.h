/*===---- __stddef_offsetof.h - Definition of offsetof ---------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(offsetof) || __has_feature(modules)
/* Always define offsetof when modules are available. */
#define offsetof(t, d) __builtin_offsetof(t, d)
#endif
