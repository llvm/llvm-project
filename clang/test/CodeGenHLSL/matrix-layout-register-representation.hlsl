// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library \
// RUN:   -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - %s | FileCheck %s

export float2x2 load_row_major(row_major float2x2 matrix) {
  return matrix;
}

// CHECK-LABEL: define {{.*}} <4 x float> @_Z14load_row_major
// CHECK: [[TO_MEMORY:%.*]] = call {{.*}} <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %matrix, i32 2, i32 2)
// CHECK: store <4 x float> [[TO_MEMORY]], ptr %matrix.addr
// CHECK: [[FROM_MEMORY:%.*]] = load <4 x float>, ptr %matrix.addr
// CHECK: [[TO_REGISTER:%.*]] = call {{.*}} <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> [[FROM_MEMORY]], i32 2, i32 2)
// CHECK: ret <4 x float> [[TO_REGISTER]]

export void store_row_major(out row_major float2x3 destination,
                            column_major float2x3 source) {
  destination = source;
}

// CHECK-LABEL: define {{.*}} @_Z15store_row_major
// CHECK: [[SOURCE:%.*]] = load <6 x float>, ptr %source.addr
// CHECK: [[TO_MEMORY:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[SOURCE]], i32 2, i32 3)
// CHECK: store <6 x float> [[TO_MEMORY]], ptr %{{.*}}

export float2x2 multiply(row_major float2x3 lhs,
                         column_major float3x2 rhs) {
  return mul(lhs, rhs);
}

// CHECK-LABEL: define {{.*}} <4 x float> @_Z8multiply
// CHECK: [[LHS_MEMORY:%.*]] = load <6 x float>, ptr %lhs.addr
// CHECK: [[LHS_REGISTER:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[LHS_MEMORY]], i32 3, i32 2)
// CHECK: [[RHS_REGISTER:%.*]] = load <6 x float>, ptr %rhs.addr
// CHECK-NOT: @llvm.matrix.transpose
// CHECK: call {{.*}} <4 x float> @llvm.matrix.multiply.v4f32.v6f32.v6f32(<6 x float> [[LHS_REGISTER]], <6 x float> [[RHS_REGISTER]], i32 2, i32 3, i32 2)