// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl \
// RUN:   -ast-dump=json -disable-llvm-passes -o - -hlsl-entry main %s \
// RUN:   | FileCheck %s

typedef float float4x4 __attribute__((matrix_type(4,4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

float4x4 gMat;

[numthreads(1, 1, 1)]
void main() {
  float4x4 A = gMat;

  // one-based swizzle
  float3 v1 = A._11_22_33;

  // zero-based swizzle
  float2 v2 = A._m00_m11;
}

// CHECK: "kind": "MatrixElementExpr"
// CHECK-NEXT: "range": {
// CHECK: "kind": "MatrixElementExpr"
// CHECK-NEXT: "range": {
