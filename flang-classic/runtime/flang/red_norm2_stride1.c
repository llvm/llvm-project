/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* red_norm2_stride1.c -- intrinsic reduction function */

#include "norm2.h"
#include <math.h>

void NORM2_REAL4 (__POINT_T *src_pointer, __INT_T *size, __REAL4_T *result) {
  //  Passing in integer*8 address of starting point in the array
  __REAL4_T *src = (__REAL4_T *)(*src_pointer);
  __REAL8_T sum = 0, val;
  __INT_T i;

  for (i = 0; i < *size; ++i) {
    val = (__REAL8_T) src[i];
    sum += val*val;
  }
  val = sqrt(sum);
  *result = (__REAL4_T) val;
}

void NORM2_REAL8 (__POINT_T *src_pointer, __INT_T *size, __REAL8_T *result) {
  // Passing in integer*8 address of starting point in the array
  __REAL8_T *src = (__REAL8_T *)(*src_pointer);
  __REAL8_T sum = 0;
  __INT_T i;

  for (i = 0; i < *size; ++i) {
    sum += src[i]*src[i];
  }
  *result = sqrt(sum);
}
