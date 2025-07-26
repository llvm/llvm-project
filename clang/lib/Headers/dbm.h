/*===---- dbm.h - BSD header for database management ----------------------===*\
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
\*===----------------------------------------------------------------------===*/

#if !defined(_AIX)

#include_next <dbm.h>

#else

#define __need_NULL
#include <stddef.h>

#include_next <dbm.h>

/* Ensure that the definition of NULL is as expected. */
#define __need_NULL
#include <stddef.h>

#endif
