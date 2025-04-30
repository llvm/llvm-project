/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	exit3f.c - Implements LIB3F exit subprogram.  */

#include "ent3f.h"
#include "enames.h"
void __fort_exit(int);

void ENT3F(EXIT, exit)(int *s) { Ftn_exit(*s); }
