/*===-- include/flang/Runtime/entry-names.h -------------------------*- C -*-=//
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
 */

/* Defines the macro RTNAME(n) which decorates the external name of a runtime
 * library function or object with extra characters so that it
 * (a) is not in the user's name space,
 * (b) doesn't conflict with other libraries, and
 * (c) prevents incompatible versions of the runtime library from linking
 *
 * The value of REVISION should not be changed until/unless the API to the
 * runtime library must change in some way that breaks backward compatibility.
 */
#ifndef RTNAME
#define NAME_WITH_PREFIX_AND_REVISION(prefix, revision, name) \
  prefix##revision##name
#define RTNAME(name) NAME_WITH_PREFIX_AND_REVISION(_Fortran, A, name)
#endif
