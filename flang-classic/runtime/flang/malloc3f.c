/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	malloc3f.c - Implements LIB3F malloc subprogram.  */

#include "fortDt.h"
#include "mpalloc.h"
#include "ent3f.h"

__POINT_T
ENT3F(MALLOC, malloc)(int *n) { return (__POINT_T)_mp_malloc((__POINT_T)(*n)); }
