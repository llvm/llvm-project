// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s


// CHECK: define {{.*}} float {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// CHECK: [[VALRET:%.*]] = tail call { i32, i32 } @llvm.dx.splitdouble.i32(double [[VALD]])
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 0
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 1
float test_scalar(double D) {
  uint A, B;
  asuint(D, A, B);
  return A + B;
}

// CHECK: define {{.*}} <3 x float> {{.*}}test_vector{{.*}}(<3 x double> {{.*}} [[VALD:%.*]])
// CHECK-COUNT-3: [[VALREG:%.*]] = extractelement <3 x double> [[VALD]], i64 [[VALIDX:[0-3]]]
// CHECK-NEXT: [[VALRET:%.*]] = tail call { i32, i32 } @llvm.dx.splitdouble.i32(double [[VALREG]])
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 0
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 1
float3 test_vector(double3 D) {
  uint3 A, B;
  asuint(D, A, B);
  return A + B;
}
