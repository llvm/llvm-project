// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

float cosf(float f) {
  return __builtin_cosf(f);
  // CIR: %{{.*}} = cir.cos %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.cos.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.cos.f32(float %{{.*}})
}

double cos(double f) {
  return __builtin_cos(f);
  // CIR: %{{.*}} = cir.cos %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.cos.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.cos.f64(double %{{.*}})
}

float ceil(float f) {
  return __builtin_ceilf(f);
  // CIR: %{{.*}} = cir.ceil %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.ceil.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.ceil.f32(float %{{.*}})
}

float expf(float f) {
  return __builtin_expf(f);
  // CIR: %{{.*}} = cir.exp {{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.exp.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.exp.f32(float %{{.*}})
}

double exp(double f) {
  return __builtin_exp(f);
  // CIR: %{{.*}} = cir.exp {{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.exp.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.exp.f64(double %{{.*}})
}

long double expl(long double f) {
  return __builtin_expl(f);
  // CIR: %{{.*}} = cir.exp {{.*}} : !cir.long_double<!cir.f128>
  // LLVM: %{{.*}} = call fp128 @llvm.exp.f128(fp128 %{{.*}})
  // OGCG: %{{.*}} = call fp128 @llvm.exp.f128(fp128 %{{.*}})
}
