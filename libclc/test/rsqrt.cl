#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void foo(float4 *x, double4 *y) {
  x[1] = rsqrt(x[0]);
  y[1] = rsqrt(y[0]);
}
