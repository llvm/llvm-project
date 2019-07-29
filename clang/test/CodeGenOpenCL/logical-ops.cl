// RUN: %clang_cc1 %s -emit-llvm -o - -cl-std=CL1.2 -O1 -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -cl-std=clc++ -O1 -triple x86_64-unknown-linux-gnu | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef int int4 __attribute((ext_vector_type(4)));
typedef long long4 __attribute((ext_vector_type(4)));
typedef float float4 __attribute((ext_vector_type(4)));
typedef double double4 __attribute((ext_vector_type(4)));

// CHECK: floatops
kernel void floatops(global int4 *out, global float4 *fout) {
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  out[0] = (float4)(1, 1, 1, 1) && 1.0f;
  // CHECK: store <4 x i32> zeroinitializer
  out[1] = (float4)(0, 0, 0, 0) && (float4)(0, 0, 0, 0);

  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  out[2] = (float4)(0, 0, 0, 0) || (float4)(1, 1, 1, 1);
  // CHECK: store <4 x i32> zeroinitializer
  out[3] = (float4)(0, 0, 0, 0) || 0.0f;

  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  out[4] = !(float4)(0, 0, 0, 0);
  // CHECK: store <4 x i32> zeroinitializer
  out[5] = !(float4)(1, 2, 3, 4);
  // CHECK: store <4 x i32> <i32 -1, i32 0, i32 -1, i32 0>
  out[6] = !(float4)(0, 1, 0, 1);
  // CHECK: store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  fout[0] = (float4)(!0.0f);
  // CHECK: store <4 x float> zeroinitializer
  fout[1] = (float4)(!1.0f);
}

// CHECK: doubleops
kernel void doubleops(global long4 *out, global double4 *dout) {
  // CHECK: store <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>
  out[0] = (double4)(1, 1, 1, 1) && 1.0;
  // CHECK: store <4 x i64> zeroinitializer
  out[1] = (double4)(0, 0, 0, 0) && (double4)(0, 0, 0, 0);

  // CHECK: store <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>
  out[2] = (double4)(0, 0, 0, 0) || (double4)(1, 1, 1, 1);
  // CHECK: store <4 x i64> zeroinitializer
  out[3] = (double4)(0, 0, 0, 0) || 0.0f;

  // CHECK: store <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>
  out[4] = !(double4)(0, 0, 0, 0);
  // CHECK: store <4 x i64> zeroinitializer
  out[5] = !(double4)(1, 2, 3, 4);
  // CHECK: store <4 x i64> <i64 -1, i64 0, i64 -1, i64 0>
  out[6] = !(double4)(0, 1, 0, 1);
  // CHECK: store <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  dout[0] = (double4)(!0.0f);
  // CHECK: store <4 x double> zeroinitializer
  dout[1] = (double4)(!1.0f);
}
