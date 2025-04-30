/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getpid3f.c - Implements LIB3F getpid subprogram.  */

#include "ent3f.h"
#if !defined(_WIN64)
#include <unistd.h>
#else
#include <process.h>
#endif
int ENT3F(GETPID, getpid)() { return getpid(); }
