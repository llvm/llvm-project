// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

float cosf(float f) {
  return __builtin_cosf(f);
  // CHECK: %{{.*}} = cir.cos {{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.cos.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.cos.f32(float %{{.*}})
}

double cos(double f) {
  return __builtin_cos(f);
  // CIR: {{.+}} = cir.cos {{.+}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.cos.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.cos.f64(double %{{.*}})
}
