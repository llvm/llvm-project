/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getgid3f.c - Implements LIB3F getgid subprogram.  */

#if !defined(_WIN64)

#include  <unistd.h>
#include "ent3f.h"

int ENT3F(GETGID, getgid)() { return getgid(); }

#endif /* !_WIN64 */
