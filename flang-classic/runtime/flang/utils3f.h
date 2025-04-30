/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef FLANG_RUNTIME_UTILS3F_H
#define FLANG_RUNTIME_UTILS3F_H

#include <stdio.h>
#include "global.h"

/* String conversion/copy between Fortran and C */
char *__fstr2cstr(char *from, int from_len);
void __cstr_free(char *from);
void __fcp_cstr(char *to, int to_len, const char *from);

/* IO-related routines needed to support 3f routines */
bool __isatty3f(int unit);
FILE *__getfile3f(int unit);

#endif /* FLANG_RUNTIME_UTILS3F_H */
