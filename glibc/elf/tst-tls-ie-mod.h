/* Module with specified TLS size and model.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* This file is parameterized by macros N, SIZE and MODEL.  */

#include <stdio.h>
#include <string.h>

#define CONCATX(x, y) x ## y
#define CONCAT(x, y) CONCATX (x, y)
#define STRX(x) #x
#define STR(x) STRX (x)

#define VAR CONCAT (var, N)

__attribute__ ((aligned (8), tls_model (MODEL)))
__thread char VAR[SIZE];

void
CONCAT (access, N) (void)
{
  printf (STR (VAR) "[%d]:\t %p .. %p " MODEL "\n", SIZE, VAR, VAR + SIZE);
  fflush (stdout);
  memset (VAR, 1, SIZE);
}
