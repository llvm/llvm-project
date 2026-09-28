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
// ROW: [[TMP:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)
// ROW-NEXT: store <6 x float> [[TMP]], ptr
// CHECK: load <6 x float>, ptr
// COL-NEXT: extractelement <6 x float> {{.*}}, i32 4
// ROW-NEXT: extractelement <6 x float> {{.*}}, i32 3
export float getScalarElement(float3x2 M) {
  return M._22;
}

// CHECK-LABEL: define {{.*}} @_Z18getSwizzleElementsu11matrix_typeILm3ELm2EfE
// ROW: [[TMP:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)
// ROW-NEXT: store <6 x float> [[TMP]], ptr
// CHECK: load <6 x float>, ptr
// COL-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 3, i32 1, i32 4>
// ROW-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
export float4 getSwizzleElements(float3x2 M) {
  return M._11_12_21_22;
}

// CHECK-LABEL: define {{.*}} @_Z22getZeroBasedSwizzleEltu11matrix_typeILm3ELm2EfE
// ROW: [[TMP:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)
// ROW-NEXT: store <6 x float> [[TMP]], ptr
// CHECK: load <6 x float>, ptr
// COL-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <2 x i32> <i32 1, i32 3>
// ROW-NEXT: shufflevector <6 x float> {{.*}}, <6 x float> poison, <2 x i32> <i32 2, i32 1>
export float2 getZeroBasedSwizzleElt(float3x2 M) {
  return M._m10_m01;
}

// transpose(m) produces a canonical column-major register value. Matrix
// element access materializes that prvalue in a matrix-typed temporary, which
// must use the selected memory layout.
export float swizzle_prvalue(float2x3 m) {
  return transpose(m)._m01;
}

// COL-LABEL: define {{.*}} float @_Z15swizzle_prvalue
// COL: [[TEMP:%.*]] = alloca [2 x <3 x float>]
// COL: [[RESULT_COL_MAJOR:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 2, i32 3)
// COL: store <6 x float> [[RESULT_COL_MAJOR]], ptr [[TEMP]]
// COL: [[FROM_TEMP:%.*]] = load <6 x float>, ptr [[TEMP]]
// COL: extractelement <6 x float> [[FROM_TEMP]], i32 3

// ROW-LABEL: define {{.*}} float @_Z15swizzle_prvalue
// ROW: [[TEMP:%.*]] = alloca [3 x <2 x float>]
// ROW: [[INPUT_COL_MAJOR:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)
// ROW: [[RESULT_COL_MAJOR:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[INPUT_COL_MAJOR]], i32 2, i32 3)
// ROW: [[TMP:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[RESULT_COL_MAJOR]], i32 3, i32 2)
// ROW: store <6 x float> [[TMP]], ptr [[TEMP]]
// ROW: [[FROM_TEMP:%.*]] = load <6 x float>, ptr [[TEMP]]
// ROW: extractelement <6 x float> [[FROM_TEMP]], i32 1
