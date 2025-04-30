/* Measure strtod implementation.
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

#define TEST_MAIN
#define TEST_NAME "strtod"

#include <stdio.h>
#include <stdlib.h>
#include "bench-timing.h"

#undef INNER_LOOP_ITERS
#define INNER_LOOP_ITERS 131072

static const char *inputs[] =
{
  "1e308",
  "100000000e300",
  "0x1p1023",
  "0x1000p1011",
  "0x1p1020",
  "0x0.00001p1040" "1e-307",
  "0.000001e-301",
  "0.0000001e-300",
  "0.00000001e-299",
  "1000000e-313",
  "10000000e-314",
  "100000000e-315",
  "0x1p-1021",
  "0x1000p-1033",
  "0x10000p-1037",
  "0x0.001p-1009",
  "0x0.0001p-1005",
  "12.345",
  "12.345e19",
  "-.1e+9",
  ".125",
  "1e20",
  "0e-19",
  "4\00012",
  "5.9e-76",
  "0x1.4p+3",
  "0xAp0",
  "0x0Ap0",
  "0x0A",
  "0xA0",
  "0x0.A0p8",
  "0x0.50p9",
  "0x0.28p10",
  "0x0.14p11",
  "0x0.0A0p12",
  "0x0.050p13",
  "0x0.028p14",
  "0x0.014p15",
  "0x00.00A0p16",
  "0x00.0050p17",
  "0x00.0028p18",
  "0x00.0014p19",
  "0x1p-1023",
  "0x0.8p-1022",
  "Inf",
  "-Inf",
  "+InFiNiTy",
  "0x80000Ap-23",
  "1e-324",
  "0x100000000000008p0",
  "0x100000000000008.p0",
  "0x100000000000008.00p0",
  "0x10000000000000800p0",
  "0x10000000000000801p0",
  NULL
};

int
do_bench (void)
{
  const size_t iters = INNER_LOOP_ITERS;

  for (size_t i = 0; inputs[i] != NULL; ++i)
    {
      char *ep;
      timing_t start, stop, cur;

      printf ("Input %-24s:", inputs[i]);
      TIMING_NOW (start);
      for (size_t j = 0; j < iters; ++j)
	strtod (inputs[i], &ep);
      TIMING_NOW (stop);

      TIMING_DIFF (cur, start, stop);
      TIMING_PRINT_MEAN ((double) cur, (double) iters);
      putchar ('\n');
    }

  return 0;
}

#define TEST_FUNCTION do_bench ()

#include "../test-skeleton.c"
