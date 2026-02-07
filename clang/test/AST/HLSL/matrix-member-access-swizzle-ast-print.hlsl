// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl \
// RUN:   -ast-print -disable-llvm-passes -o - -hlsl-entry main %s \
// RUN:   | FileCheck %s

typedef float float4x4 __attribute__((matrix_type(4,4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

float4x4 gMat;

[numthreads(1, 1, 1)]
void main() {
  float4x4 A = gMat;
  float3 v1 = A._12_21_32;
  float2 v2 = A._m01_m10;
}

// CHECK: float4x4 gMat;
// CHECK: float4x4 A = gMat;
// CHECK: float3 v1 = A._12_21_32;
// CHECK: float2 v2 = A._m01_m10;
