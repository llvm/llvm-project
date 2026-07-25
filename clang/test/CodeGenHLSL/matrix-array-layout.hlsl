// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

// Regression test for the in-memory layout of arrays of matrices with explicit
// row_major / column_major keywords.
//
// ConvertTypeForMem lays a matrix out as [ArrayLen x <VecLen>]:
//   row_major    float2x3 -> [2 x <3 x float>]  (2 rows of 3 columns)
//   column_major float2x3 -> [3 x <2 x float>]  (3 rows of 2 columns)
//
// The element of a matrix array must keep the same layout as the equivalent
// bare matrix.

export void f() {
  row_major    float2x3 rm_arr[2];
  column_major float2x3 cm_arr[2];
  row_major    float2x3 rm_bare;
  column_major float2x3 cm_bare;
  rm_arr[0] = rm_bare;
  cm_arr[0] = cm_bare;
}

// The array element layout matches the bare matrix layout for each orientation.
// CHECK: %rm_arr = alloca [2 x [2 x <3 x float>]], align 4
// CHECK: %cm_arr = alloca [2 x [3 x <2 x float>]], align 4
// CHECK: %rm_bare = alloca [2 x <3 x float>], align 4
// CHECK: %cm_bare = alloca [3 x <2 x float>], align 4
