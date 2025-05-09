//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void foo(float4 *x, double4 *y) {
  x[1] = rsqrt(x[0]);
  y[1] = rsqrt(y[0]);
}
