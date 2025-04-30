/* Test backtrace and backtrace_symbols: common code for examining
   backtraces.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/* Set to a non-zero value if the test fails.  */
volatile int ret;

/* Accesses to X are used to prevent optimization.  */
volatile int x;

/* Called if the test fails.  */
#define FAIL() \
  do { printf ("Failure on line %d\n", __LINE__); ret = 1; } while (0)

/* Use this attribute to prevent inlining, so that all expected frames
   are present.  */
#define NO_INLINE __attribute__ ((noinline, noclone, weak))

/* Look for a match in SYM from backtrace_symbols to NAME, a fragment
   of a function name.  Ignore the filename before '(', but presume
   that the function names are chosen so they cannot accidentally
   match the hex offset before the closing ')'. */

static inline bool
match (const char *sym, const char *name)
{
  char *p = strchr (sym, '(');
  return p != NULL && strstr (p, name) != NULL;
}
