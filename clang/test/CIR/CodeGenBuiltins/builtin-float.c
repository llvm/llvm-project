// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// frexp: float, double, long double

float test_frexpf(float x) {
  int exp;
  return __builtin_frexpf(x, &exp);
}

// CIR-LABEL: cir.func {{.*}} @test_frexpf
// CIR:         %{{.*}}, %{{.*}} = cir.frexp %{{.*}} : !cir.float -> !cir.float, !s32i
// CIR:         cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} float @test_frexpf
// LLVM:         %{{.*}} = call { float, i32 } @llvm.frexp.f32.i32(float %{{.*}})
// LLVM:         %{{.*}} = extractvalue { float, i32 } %{{.*}}, 0
// LLVM:         %{{.*}} = extractvalue { float, i32 } %{{.*}}, 1
// LLVM:         ret float

// OGCG-LABEL: define {{.*}} float @test_frexpf
// OGCG:         %{{.*}} = call { float, i32 } @llvm.frexp.f32.i32(float %{{.*}})
// OGCG:         %{{.*}} = extractvalue { float, i32 } %{{.*}}, 1
// OGCG:         %{{.*}} = extractvalue { float, i32 } %{{.*}}, 0
// OGCG:         ret float

double test_frexp(double x) {
  int exp;
  return __builtin_frexp(x, &exp);
}

// CIR-LABEL: cir.func {{.*}} @test_frexp
// CIR:         %{{.*}}, %{{.*}} = cir.frexp %{{.*}} : !cir.double -> !cir.double, !s32i
// CIR:         cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} double @test_frexp
// LLVM:         %{{.*}} = call { double, i32 } @llvm.frexp.f64.i32(double %{{.*}})
// LLVM:         %{{.*}} = extractvalue { double, i32 } %{{.*}}, 0
// LLVM:         %{{.*}} = extractvalue { double, i32 } %{{.*}}, 1
// LLVM:         ret double

// OGCG-LABEL: define {{.*}} double @test_frexp
// OGCG:         %{{.*}} = call { double, i32 } @llvm.frexp.f64.i32(double %{{.*}})
// OGCG:         %{{.*}} = extractvalue { double, i32 } %{{.*}}, 1
// OGCG:         %{{.*}} = extractvalue { double, i32 } %{{.*}}, 0
// OGCG:         ret double

long double test_frexpl(long double x) {
  int exp;
  return __builtin_frexpl(x, &exp);
}

// CIR-LABEL: cir.func {{.*}} @test_frexpl
// CIR:         %{{.*}}, %{{.*}} = cir.frexp %{{.*}} : !cir.long_double<!cir.f80> -> !cir.long_double<!cir.f80>, !s32i
// CIR:         cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} x86_fp80 @test_frexpl
// LLVM:         %{{.*}} = call { x86_fp80, i32 } @llvm.frexp.f80.i32(x86_fp80 %{{.*}})
// LLVM:         %{{.*}} = extractvalue { x86_fp80, i32 } %{{.*}}, 0
// LLVM:         %{{.*}} = extractvalue { x86_fp80, i32 } %{{.*}}, 1
// LLVM:         ret x86_fp80

// OGCG-LABEL: define {{.*}} x86_fp80 @test_frexpl
// OGCG:         %{{.*}} = call { x86_fp80, i32 } @llvm.frexp.f80.i32(x86_fp80 %{{.*}})
// OGCG:         %{{.*}} = extractvalue { x86_fp80, i32 } %{{.*}}, 1
// OGCG:         %{{.*}} = extractvalue { x86_fp80, i32 } %{{.*}}, 0
// OGCG:         ret x86_fp80

__float128 test_frexpf128(__float128 x) {
  int exp;
  return __builtin_frexpf128(x, &exp);
}

// CIR-LABEL: cir.func {{.*}} @test_frexpf128
// CIR:         %{{.*}}, %{{.*}} = cir.frexp %{{.*}} : !cir.f128 -> !cir.f128, !s32i
// CIR:         cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} fp128 @test_frexpf128
// LLVM:         %{{.*}} = call { fp128, i32 } @llvm.frexp.f128.i32(fp128 %{{.*}})
// LLVM:         %{{.*}} = extractvalue { fp128, i32 } %{{.*}}, 0
// LLVM:         %{{.*}} = extractvalue { fp128, i32 } %{{.*}}, 1
// LLVM:         ret fp128

// OGCG-LABEL: define {{.*}} fp128 @test_frexpf128
// OGCG:         %{{.*}} = call { fp128, i32 } @llvm.frexp.f128.i32(fp128 %{{.*}})
// OGCG:         %{{.*}} = extractvalue { fp128, i32 } %{{.*}}, 1
// OGCG:         %{{.*}} = extractvalue { fp128, i32 } %{{.*}}, 0
// OGCG:         ret fp128

// modf: float, double, long double

float test_modff(float x) {
  float ipart;
  return __builtin_modff(x, &ipart);
}

// CIR-LABEL: cir.func {{.*}} @test_modff
// CIR:         %{{.*}}, %{{.*}} = cir.modf %{{.*}} : !cir.float -> !cir.float, !cir.float
// CIR:         cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} float @test_modff
// LLVM:         %{{.*}} = call { float, float } @llvm.modf.f32(float %{{.*}})
// LLVM:         %{{.*}} = extractvalue { float, float } %{{.*}}, 0
// LLVM:         %{{.*}} = extractvalue { float, float } %{{.*}}, 1
// LLVM:         ret float

// OGCG-LABEL: define {{.*}} float @test_modff
// OGCG:         %{{.*}} = call { float, float } @llvm.modf.f32(float %{{.*}})
// OGCG:         %{{.*}} = extractvalue { float, float } %{{.*}}, 0
// OGCG:         %{{.*}} = extractvalue { float, float } %{{.*}}, 1
// OGCG:         ret float

double test_modf(double x) {
  double ipart;
  return __builtin_modf(x, &ipart);
}

// CIR-LABEL: cir.func {{.*}} @test_modf
// CIR:         %{{.*}}, %{{.*}} = cir.modf %{{.*}} : !cir.double -> !cir.double, !cir.double
// CIR:         cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} double @test_modf
// LLVM:         %{{.*}} = call { double, double } @llvm.modf.f64(double %{{.*}})
// LLVM:         %{{.*}} = extractvalue { double, double } %{{.*}}, 0
// LLVM:         %{{.*}} = extractvalue { double, double } %{{.*}}, 1
// LLVM:         ret double

// OGCG-LABEL: define {{.*}} double @test_modf
// OGCG:         %{{.*}} = call { double, double } @llvm.modf.f64(double %{{.*}})
// OGCG:         %{{.*}} = extractvalue { double, double } %{{.*}}, 0
// OGCG:         %{{.*}} = extractvalue { double, double } %{{.*}}, 1
// OGCG:         ret double

long double test_modfl(long double x) {
  long double ipart;
  return __builtin_modfl(x, &ipart);
}

// CIR-LABEL: cir.func {{.*}} @test_modfl
// CIR:         %{{.*}}, %{{.*}} = cir.modf %{{.*}} : !cir.long_double<!cir.f80> -> !cir.long_double<!cir.f80>, !cir.long_double<!cir.f80>
// CIR:         cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} x86_fp80 @test_modfl
// LLVM:         %{{.*}} = call { x86_fp80, x86_fp80 } @llvm.modf.f80(x86_fp80 %{{.*}})
// LLVM:         %{{.*}} = extractvalue { x86_fp80, x86_fp80 } %{{.*}}, 0
// LLVM:         %{{.*}} = extractvalue { x86_fp80, x86_fp80 } %{{.*}}, 1
// LLVM:         ret x86_fp80

// OGCG-LABEL: define {{.*}} x86_fp80 @test_modfl
// OGCG:         %{{.*}} = call { x86_fp80, x86_fp80 } @llvm.modf.f80(x86_fp80 %{{.*}})
// OGCG:         %{{.*}} = extractvalue { x86_fp80, x86_fp80 } %{{.*}}, 0
// OGCG:         %{{.*}} = extractvalue { x86_fp80, x86_fp80 } %{{.*}}, 1
// OGCG:         ret x86_fp80

__float128 test_modff128(__float128 x) {
  __float128 ipart;
  return __builtin_modff128(x, &ipart);
}

// CIR-LABEL: cir.func {{.*}} @test_modff128
// CIR:         cir.call @modff128
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} fp128 @test_modff128
// LLVM:         call fp128 @modff128(fp128 {{.*}}, ptr {{.*}})
// LLVM:         ret fp128

// OGCG-LABEL: define {{.*}} fp128 @test_modff128
// OGCG:         call fp128 @modff128(fp128 {{.*}}, ptr {{.*}})
// OGCG:         ret fp128

// powi: float, double, long double

float test_powif(float x, int y) {
  return __builtin_powif(x, y);
}

// CIR-LABEL: cir.func {{.*}} @test_powif
// CIR:         %{{.*}} = cir.call_llvm_intrinsic "powi" %{{.*}}, %{{.*}} : (!cir.float, !s32i) -> !cir.float
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} float @test_powif
// LLVM:         call float @llvm.powi.f32.i32(float %{{.*}}, i32 %{{.*}})
// LLVM:         ret float

// OGCG-LABEL: define {{.*}} float @test_powif
// OGCG:         call float @llvm.powi.f32.i32(float %{{.*}}, i32 %{{.*}})
// OGCG:         ret float

double test_powi(double x, int y) {
  return __builtin_powi(x, y);
}

// CIR-LABEL: cir.func {{.*}} @test_powi
// CIR:         %{{.*}} = cir.call_llvm_intrinsic "powi" %{{.*}}, %{{.*}} : (!cir.double, !s32i) -> !cir.double
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} double @test_powi
// LLVM:         call double @llvm.powi.f64.i32(double %{{.*}}, i32 %{{.*}})
// LLVM:         ret double

// OGCG-LABEL: define {{.*}} double @test_powi
// OGCG:         call double @llvm.powi.f64.i32(double %{{.*}}, i32 %{{.*}})
// OGCG:         ret double

long double test_powil(long double x, int y) {
  return __builtin_powil(x, y);
}

// CIR-LABEL: cir.func {{.*}} @test_powil
// CIR:         %{{.*}} = cir.call_llvm_intrinsic "powi" %{{.*}}, %{{.*}} : (!cir.long_double<!cir.f80>, !s32i) -> !cir.long_double<!cir.f80>
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} x86_fp80 @test_powil
// LLVM:         call x86_fp80 @llvm.powi.f80.i32(x86_fp80 %{{.*}}, i32 %{{.*}})
// LLVM:         ret x86_fp80

// OGCG-LABEL: define {{.*}} x86_fp80 @test_powil
// OGCG:         call x86_fp80 @llvm.powi.f80.i32(x86_fp80 %{{.*}}, i32 %{{.*}})
// OGCG:         ret x86_fp80

// FP comparison builtins: float

void test_floats(float x, float y, int *out) {
  out[0] = __builtin_isgreater(x, y);
  // CIR: cir.cmp gt %{{.*}}, %{{.*}} : !cir.float
  // LLVM: fcmp ogt float
  // OGCG: fcmp ogt float

  out[1] = __builtin_isgreaterequal(x, y);
  // CIR: cir.cmp ge %{{.*}}, %{{.*}} : !cir.float
  // LLVM: fcmp oge float
  // OGCG: fcmp oge float

  out[2] = __builtin_isless(x, y);
  // CIR: cir.cmp lt %{{.*}}, %{{.*}} : !cir.float
  // LLVM: fcmp olt float
  // OGCG: fcmp olt float

  out[3] = __builtin_islessequal(x, y);
  // CIR: cir.cmp le %{{.*}}, %{{.*}} : !cir.float
  // LLVM: fcmp ole float
  // OGCG: fcmp ole float

  out[4] = __builtin_islessgreater(x, y);
  // CIR: cir.cmp one %{{.*}}, %{{.*}} : !cir.float
  // LLVM: fcmp one float
  // OGCG: fcmp one float

  out[5] = __builtin_isunordered(x, y);
  // CIR: cir.cmp uno %{{.*}}, %{{.*}} : !cir.float
  // LLVM: fcmp uno float
  // OGCG: fcmp uno float
}

// FP comparison builtins: double

void test_doubles(double x, double y, int *out) {
  out[0] = __builtin_isgreater(x, y);
  // CIR: cir.cmp gt %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fcmp ogt double
  // OGCG: fcmp ogt double

  out[1] = __builtin_isgreaterequal(x, y);
  // CIR: cir.cmp ge %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fcmp oge double
  // OGCG: fcmp oge double

  out[2] = __builtin_isless(x, y);
  // CIR: cir.cmp lt %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fcmp olt double
  // OGCG: fcmp olt double

  out[3] = __builtin_islessequal(x, y);
  // CIR: cir.cmp le %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fcmp ole double
  // OGCG: fcmp ole double

  out[4] = __builtin_islessgreater(x, y);
  // CIR: cir.cmp one %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fcmp one double
  // OGCG: fcmp one double

  out[5] = __builtin_isunordered(x, y);
  // CIR: cir.cmp uno %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fcmp uno double
  // OGCG: fcmp uno double
}

// FP comparison builtins: mixed double + float (fpext)

void test_mixed(double x, float y, int *out) {
  out[0] = __builtin_isgreater(x, y);
  // CIR: cir.cast floating %{{.*}} : !cir.float -> !cir.double
  // CIR: cir.cmp gt %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fpext float %{{.*}} to double
  // LLVM: fcmp ogt double
  // OGCG: fpext float %{{.*}} to double
  // OGCG: fcmp ogt double

  out[1] = __builtin_isgreaterequal(x, y);
  // CIR: cir.cast floating %{{.*}} : !cir.float -> !cir.double
  // CIR: cir.cmp ge %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fpext float %{{.*}} to double
  // LLVM: fcmp oge double
  // OGCG: fpext float %{{.*}} to double
  // OGCG: fcmp oge double

  out[2] = __builtin_isless(x, y);
  // CIR: cir.cast floating %{{.*}} : !cir.float -> !cir.double
  // CIR: cir.cmp lt %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fpext float %{{.*}} to double
  // LLVM: fcmp olt double
  // OGCG: fpext float %{{.*}} to double
  // OGCG: fcmp olt double

  out[3] = __builtin_islessequal(x, y);
  // CIR: cir.cast floating %{{.*}} : !cir.float -> !cir.double
  // CIR: cir.cmp le %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fpext float %{{.*}} to double
  // LLVM: fcmp ole double
  // OGCG: fpext float %{{.*}} to double
  // OGCG: fcmp ole double

  out[4] = __builtin_islessgreater(x, y);
  // CIR: cir.cast floating %{{.*}} : !cir.float -> !cir.double
  // CIR: cir.cmp one %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fpext float %{{.*}} to double
  // LLVM: fcmp one double
  // OGCG: fpext float %{{.*}} to double
  // OGCG: fcmp one double

  out[5] = __builtin_isunordered(x, y);
  // CIR: cir.cast floating %{{.*}} : !cir.float -> !cir.double
  // CIR: cir.cmp uno %{{.*}}, %{{.*}} : !cir.double
  // LLVM: fpext float %{{.*}} to double
  // LLVM: fcmp uno double
  // OGCG: fpext float %{{.*}} to double
  // OGCG: fcmp uno double
}

// FP comparison builtins: __fp16 (promoted to float)

int test_isgreater_half(__fp16 *a, __fp16 *b) {
  return __builtin_isgreater(*a, *b);
}

// CIR-LABEL: cir.func {{.*}} @test_isgreater_half
// CIR:         cir.cast floating %{{.*}} : !cir.f16 -> !cir.float
// CIR:         cir.cast floating %{{.*}} : !cir.f16 -> !cir.float
// CIR:         cir.cmp gt %{{.*}}, %{{.*}} : !cir.float

// LLVM-LABEL: define {{.*}} i32 @test_isgreater_half
// LLVM:         fpext half %{{.*}} to float
// LLVM:         fpext half %{{.*}} to float
// LLVM:         fcmp ogt float
// LLVM:         zext i1 %{{.*}} to i32
// LLVM:         ret i32

// OGCG-LABEL: define {{.*}} i32 @test_isgreater_half
// OGCG:         fpext half %{{.*}} to float
// OGCG:         fpext half %{{.*}} to float
// OGCG:         fcmp ogt float
// OGCG:         zext i1 %{{.*}} to i32
// OGCG:         ret i32
