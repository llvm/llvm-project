// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O0 -o - | FileCheck %s

// CHECK: define {{.*}}test_scalar{{.*}}(double {{.*}} [[VAL1:%.*]], i32 {{.*}} [[VAL2:%.*]], i32 {{.*}} [[VAL3:%.*]]){{.*}} 
// CHECK: [[VALD:%.*]] = load double, ptr [[VAL1]].addr{{.*}}
// CHECK: call { i32, i32 } @llvm.dx.asuint.splitdouble.{{.*}}(double [[VALD]])
float fn(double D) {
  uint A, B;
  asuint(D, A, B);
  return A + B;
}
