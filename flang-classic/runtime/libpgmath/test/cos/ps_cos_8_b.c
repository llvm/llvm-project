
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// RUN: %libpgmath-compile -DMAX_VREG_SIZE=256 && %libpgmath-run

#define FUNC cos
#define FRP p
#define PREC s
#define VL 8
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
	VRS_T expd_res = {0.98614f, 0.98981f, 0.99220f, 0.99383f, 0.99500f, 0.99587f, 0.99653f, 0.99704f };
  VIS_T vmask __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1};
#if !defined(TARGET_WIN)
	parseargs(argc, argv);
#endif

#include "single1.h"

}

// UNSUPPORTED: sse4
// UNSUPPORTED: em64t
// UNSUPPORTED: ppc64le
// UNSUPPORTED: aarch64
