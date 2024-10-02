// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s



// CHECK: define {{.*}} i32 {{.*}}test_scalar{{.*}}(double {{.*}} [[VALD:%.*]])
// CHECK: [[VALRET:%.*]] = {{.*}} call { i32, i32 } @llvm.dx.splitdouble.i32(double [[VALD]])
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 0
// CHECK-NEXT: extractvalue { i32, i32 } [[VALRET]], 1
uint test_scalar(double D) {
  uint A, B;
  asuint(D, A, B);
  return A + B;
}


// CHECK: define {{.*}} <3 x i32> {{.*}}test_vector{{.*}}(<3 x double> {{.*}} [[VALD:%.*]])
// CHECK: [[VALRET:%.*]] = {{.*}} call { <3 x i32>, <3 x i32> } @llvm.dx.splitdouble.v3i32(<3 x double> [[VALD]])
// CHECK-NEXT: extractvalue { <3 x i32>, <3 x i32> } [[VALRET]], 0
// CHECK-NEXT: extractvalue { <3 x i32>, <3 x i32> } [[VALRET]], 1
uint3 test_vector(double3 D) {
  uint3 A, B;
  asuint(D, A, B);
  return A + B;
}
