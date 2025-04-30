/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	time3f.c - Implements LIB3F time function.  */

#include <time.h>
#include "ent3f.h"

#if defined(_WIN64)
#include "wintimes.h"
#endif

#if defined(TARGET_INTERIX_X8664) || defined(TARGET_WIN_X8664)
#define INT long long
#else
#define INT long
#endif

INT ENT3F(TIME, time)() { return time(0); }

long long ENT3F(TIME8, time8)() { return time(0); }
