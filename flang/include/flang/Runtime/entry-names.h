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
#ifndef FORTRAN_RUNTIME_ENTRY_NAMES_H
#define FORTRAN_RUNTIME_ENTRY_NAMES_H

#include "flang/Common/api-attrs.h"

#ifndef RTNAME
#define NAME_WITH_PREFIX_AND_REVISION(prefix, revision, name) \
  prefix##revision##name
#define RTNAME(name) NAME_WITH_PREFIX_AND_REVISION(_Fortran, A, name)
#endif

#ifndef RTDECL
#define RTDECL(name) RT_API_ATTRS RTNAME(name)
#endif

#ifndef RTDEF
#define RTDEF(name) RT_API_ATTRS RTNAME(name)
#endif

#ifndef RTNAME_STRING
#define RTNAME_STRINGIFY_(x) #x
#define RTNAME_STRINGIFY(x) RTNAME_STRINGIFY_(x)
#define RTNAME_STRING(name) RTNAME_STRINGIFY(RTNAME(name))
#endif

#endif /* !FORTRAN_RUNTIME_ENTRY_NAMES_H */
