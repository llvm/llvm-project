// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -disable-llvm-passes \
// RUN:   -emit-llvm -finclude-default-header -fmatrix-memory-layout=column-major \
// RUN:   -o - %s | FileCheck %s --check-prefixes=CHECK,COL
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -disable-llvm-passes \
// RUN:   -emit-llvm -finclude-default-header -fmatrix-memory-layout=row-major \
// RUN:   -o - %s | FileCheck %s --check-prefixes=CHECK,ROW

// For a float3x2 matrix (3 rows, 2 columns):
//   Column-major flat vector: [_11, _21, _31, _12, _22, _32]
//                         idx:  0    1    2    3    4    5
//   Row-major flat vector:    [_11, _12, _21, _22, _31, _32]
//                         idx:  0    1    2    3    4    5


// CHECK-LABEL: define {{.*}} @_Z16getScalarElementu11matrix_typeILm3ELm2EfE
// CHECK: load <6 x float>, ptr
// COL-NEXT: extractelement <6 x float> {{.*}}, i32 4
// ROW-NEXT: extractelement <6 x float> {{.*}}, i32 3
export float getScalarElement(float3x2 M) {
  return M._22;
}

// CHECK-LABEL: define {{.*}} @_Z18getSwizzleElementsu11matrix_typeILm3ELm2EfE
// CHECK: load <6 x float>, ptr
// COL-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 3, i32 1, i32 4>
// ROW-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
export float4 getSwizzleElements(float3x2 M) {
  return M._11_12_21_22;
}

// CHECK-LABEL: define {{.*}} @_Z22getZeroBasedSwizzleEltu11matrix_typeILm3ELm2EfE
// CHECK: load <6 x float>, ptr
// COL-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <2 x i32> <i32 1, i32 3>
// ROW-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <2 x i32> <i32 2, i32 1>
export float2 getZeroBasedSwizzleElt(float3x2 M) {
  return M._m10_m01;
}
