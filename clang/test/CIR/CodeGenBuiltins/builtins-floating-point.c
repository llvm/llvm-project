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

float exp2f(float f) {
  return __builtin_exp2f(f);
  // CIR: %{{.*}} = cir.exp2 {{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.exp2.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.exp2.f32(float %{{.*}})
}

double my_exp2(double f) {
  return __builtin_exp2(f);
  // CIR: %{{.*}} = cir.exp2 {{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.exp2.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.exp2.f64(double %{{.*}})
}

long double my_exp2l(long double f) {
  return __builtin_exp2l(f);
  // CIR: %{{.*}} = cir.exp2 {{.*}} : !cir.long_double<!cir.f128>
  // LLVM: %{{.*}} = call fp128 @llvm.exp2.f128(fp128 %{{.*}})
  // OGCG: %{{.*}} = call fp128 @llvm.exp2.f128(fp128 %{{.*}})
}

float floorf(float f) {
  return __builtin_floorf(f);
  // CIR: %{{.*}} = cir.floor %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.floor.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.floor.f32(float %{{.*}})
}

double floor(double f) {
  return __builtin_floor(f);
  // CIR: %{{.*}} = cir.floor %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.floor.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.floor.f64(double %{{.*}})
}

long double floorl(long double f) {
  return __builtin_floorl(f);
  // CIR: %{{.*}} = cir.floor %{{.*}} : !cir.long_double<!cir.f128>
  // LLVM: %{{.*}} = call fp128 @llvm.floor.f128(fp128 %{{.*}})
  // OGCG: %{{.*}} = call fp128 @llvm.floor.f128(fp128 %{{.*}})
}
