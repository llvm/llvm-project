// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK-LABEL: define {{.*}}test_transpose_bool2x3
// CHECK:    [[A_ADDR:%.*]] = alloca [3 x <2 x i32>], align 4
// CHECK:    [[A_EXT:%.*]] = zext <6 x i1> %{{.*}} to <6 x i32>
// CHECK:    store <6 x i32> [[A_EXT]], ptr [[A_ADDR]], align 4
// CHECK:    [[A:%.*]] = load <6 x i32>, ptr [[A_ADDR]], align 4
// CHECK:    [[TRANS:%.*]] = call <6 x i32> @llvm.matrix.transpose.v6i32(<6 x i32> [[A]], i32 2, i32 3)
bool3x2 test_transpose_bool2x3(bool2x3 a) {
  return transpose(a);
}

// CHECK-LABEL: define {{.*}}test_transpose_int4x3
// CHECK:    [[A_ADDR:%.*]] = alloca [3 x <4 x i32>], align 4
// CHECK:    store <12 x i32> %{{.*}}, ptr [[A_ADDR]], align 4
// CHECK:    [[A:%.*]] = load <12 x i32>, ptr [[A_ADDR]], align 4
// CHECK:    [[TRANS:%.*]] = call <12 x i32> @llvm.matrix.transpose.v12i32(<12 x i32> [[A]], i32 4, i32 3)
// CHECK:    ret <12 x i32> [[TRANS]]
int3x4 test_transpose_int4x3(int4x3 a) {
  return transpose(a);
}

// CHECK-LABEL: define {{.*}}test_transpose_float4x4
// CHECK:    [[A_ADDR:%.*]] = alloca [4 x <4 x float>], align 4
// CHECK:    store <16 x float> %{{.*}}, ptr [[A_ADDR]], align 4
// CHECK:    [[A:%.*]] = load <16 x float>, ptr [[A_ADDR]], align 4
// CHECK:    [[TRANS:%.*]] = call {{.*}}<16 x float> @llvm.matrix.transpose.v16f32(<16 x float> [[A]], i32 4, i32 4)
// CHECK:    ret <16 x float> [[TRANS]]
float4x4 test_transpose_float4x4(float4x4 a) {
  return transpose(a);
}

// CHECK-LABEL: define {{.*}}test_transpose_double1x4
// CHECK:    [[A_ADDR:%.*]] = alloca [4 x <1 x double>], align 8
// CHECK:    store <4 x double> %{{.*}}, ptr [[A_ADDR]], align 8
// CHECK:    [[A:%.*]] = load <4 x double>, ptr [[A_ADDR]], align 8
// CHECK:    [[TRANS:%.*]] = call {{.*}}<4 x double> @llvm.matrix.transpose.v4f64(<4 x double> [[A]], i32 1, i32 4)
// CHECK:    ret <4 x double> [[TRANS]]
double4x1 test_transpose_double1x4(double1x4 a) {
  return transpose(a);
}
