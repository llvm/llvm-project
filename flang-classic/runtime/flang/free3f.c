/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	free3f.c - Implements LIB3F free subprogram.  */

#include "ent3f.h"
#include "mpalloc.h"

void ENT3F(FREE, free)(char **p) { _mp_free(*p); }
