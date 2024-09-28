// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv--vulkan-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=SPIRV



// CHECK: define {{.*}} float {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// CHECK: [[VALRET:%hlsl.asuint.*]] = {{.*}} call { i32, i32 } @llvm.dx.splitdouble.i32(double [[VALD]])
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 0
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 1
// SPIRV: define spir_func {{.*}} float {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// SPIRV-NOT: @llvm.dx.splitdouble
// SPIRV: [[REG:%.*]] = load double, ptr [[VALD]].addr
// SPIRV: call spir_func void {{.*}}asuint{{.*}}(double {{.*}} [[REG]], {{.*}})
float test_scalar(double D) {
  uint A, B;
  asuint(D, A, B);
  return A + B;
}


// CHECK: define {{.*}} <3 x float> {{.*}}test_vector{{.*}}(<3 x double> {{.*}} [[VALD:%.*]])
// CHECK-COUNT-3: [[VALREG:%.*]] = extractelement <3 x double> [[VALD]], i64 [[VALIDX:[0-3]]]
// CHECK-NEXT: [[VALRET:%hlsl.asuint.*]] = {{.*}} call { i32, i32 } @llvm.dx.splitdouble.i32(double [[VALREG]])
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 0
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 1
// SPIRV: define spir_func {{.*}} <3 x float> {{.*}}test_vector{{.*}}(<3 x double> {{.*}} [[VALD:%.*]])
// SPIRV-NOT: @llvm.dx.splitdouble
// SPIRV: [[REG:%.*]] = load <3 x double>, ptr [[VALD]].addr
// SPIRV: call spir_func void {{.*}}asuint{{.*}}(<3 x double> {{.*}} [[REG]], {{.*}})
float3 test_vector(double3 D) {
  uint3 A, B;
  asuint(D, A, B);
  return A + B;
}
