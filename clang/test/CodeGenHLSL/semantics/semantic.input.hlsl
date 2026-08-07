// Per-row load.input emission for the range of semantic leaf types:
//   float          - scalar        -> 1 row
//   float4         - vector        -> 1 row (4 columns)
//   float[5]       - scalar array  -> 5 rows
//   float4[2][3]   - vector array  -> 6 rows (multidimensional, 4 columns each)
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  float a       : A;
  float4 b      : B;
  float d[5]    : D;
  float4 e[2][3] : E;
};

[shader("vertex")]
void main(S s) {}

// float a : A -> 1 row, 1 column.
// CHECK: %[[A:.*]] = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
// CHECK: %[[S0:.*]] = insertvalue %struct.S poison, float %[[A]], 0

// float4 b : B -> 1 row, 4 columns.
// CHECK: %[[B:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 1, i32 0, i8 0, i32 poison)
// CHECK: %[[S1:.*]] = insertvalue %struct.S %[[S0]], <4 x float> %[[B]], 1

// float d[5] : D -> 5 rows, 1 column each.
// CHECK: %[[D0:.*]] = call float @llvm.dx.load.input.f32(i32 2, i32 0, i8 0, i32 poison)
// CHECK: %[[D_ARRAY0:.*]] = insertvalue [5 x float] poison, float %[[D0]], 0
// CHECK: %[[D1:.*]] = call float @llvm.dx.load.input.f32(i32 2, i32 1, i8 0, i32 poison)
// CHECK: %[[D_ARRAY1:.*]] = insertvalue [5 x float] %[[D_ARRAY0]], float %[[D1]], 1
// CHECK: %[[D2:.*]] = call float @llvm.dx.load.input.f32(i32 2, i32 2, i8 0, i32 poison)
// CHECK: %[[D_ARRAY2:.*]] = insertvalue [5 x float] %[[D_ARRAY1]], float %[[D2]], 2
// CHECK: %[[D3:.*]] = call float @llvm.dx.load.input.f32(i32 2, i32 3, i8 0, i32 poison)
// CHECK: %[[D_ARRAY3:.*]] = insertvalue [5 x float] %[[D_ARRAY2]], float %[[D3]], 3
// CHECK: %[[D4:.*]] = call float @llvm.dx.load.input.f32(i32 2, i32 4, i8 0, i32 poison)
// CHECK: %[[D_ARRAY4:.*]] = insertvalue [5 x float] %[[D_ARRAY3]], float %[[D4]], 4
// CHECK: %[[S2:.*]] = insertvalue %struct.S %[[S1]], [5 x float] %[[D_ARRAY4]], 2

// float4 e[2][3] : E -> 6 rows (2 x 3), 4 columns each; row-major flattening.
// CHECK: %[[E0:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 3, i32 0, i8 0, i32 poison)
// CHECK: %[[E_ARRAY0:.*]] = insertvalue [2 x [3 x <4 x float>]] poison, <4 x float> %[[E0]], 0, 0
// CHECK: %[[E1:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 3, i32 1, i8 0, i32 poison)
// CHECK: %[[E_ARRAY1:.*]] = insertvalue [2 x [3 x <4 x float>]] %[[E_ARRAY0]], <4 x float> %[[E1]], 0, 1
// CHECK: %[[E2:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 3, i32 2, i8 0, i32 poison)
// CHECK: %[[E_ARRAY2:.*]] = insertvalue [2 x [3 x <4 x float>]] %[[E_ARRAY1]], <4 x float> %[[E2]], 0, 2
// CHECK: %[[E3:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 3, i32 3, i8 0, i32 poison)
// CHECK: %[[E_ARRAY3:.*]] = insertvalue [2 x [3 x <4 x float>]] %[[E_ARRAY2]], <4 x float> %[[E3]], 1, 0
// CHECK: %[[E4:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 3, i32 4, i8 0, i32 poison)
// CHECK: %[[E_ARRAY4:.*]] = insertvalue [2 x [3 x <4 x float>]] %[[E_ARRAY3]], <4 x float> %[[E4]], 1, 1
// CHECK: %[[E5:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 3, i32 5, i8 0, i32 poison)
// CHECK: %[[E_ARRAY5:.*]] = insertvalue [2 x [3 x <4 x float>]] %[[E_ARRAY4]], <4 x float> %[[E5]], 1, 2
// CHECK: %[[S3:.*]] = insertvalue %struct.S %[[S2]], [2 x [3 x <4 x float>]] %[[E_ARRAY5]], 3
