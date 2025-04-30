/* Check IFUNC resolver with CPU_FEATURE_USABLE and tunables.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include "tst-ifunc-isa.h"
#include <support/test-driver.h>

static int
do_test (void)
{
  /* CPU must support SSE2.  */
  if (!__builtin_cpu_supports ("sse2"))
    return EXIT_UNSUPPORTED;
  enum isa value = foo ();
  /* All ISAs, but SSE2, are disabled by tunables.  */
  return value == sse2 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include <support/test-driver.c>
