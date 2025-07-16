/*===---- stdio.h - Standard header for input and output-------------------===*\
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
\*===----------------------------------------------------------------------===*/

#if !defined(_AIX)

#include_next <stdio.h>

#else

#define __need_NULL
#include <stddef.h>

#include_next <stdio.h>

/* Ensure that the definition of NULL is as expected. */
#define __need_NULL
#include <stddef.h>

#endif
