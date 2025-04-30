
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// RUN: %libpgmath-compile -DMAX_VREG_SIZE=64 && %libpgmath-run

#define FUNC cos
#define FRP r
#define PREC s
#define VL 1
#define TOL 0.00001f

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#if !defined(TARGET_WIN)
#include <unistd.h>
#endif

#include "pgmath_test.h"

int main(int argc, char *argv[])
{
	VRS_T expd_res = 0.54030f;
  VIS_T vmask __attribute__((aligned(64))) = {-1};
#if !defined(TARGET_WIN)
	parseargs(argc, argv);
#endif

#include "single1.h"

}

