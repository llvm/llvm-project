// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-apple-darwin-macho -fclangir -emit-cir %s -o %t-aarch64.cir
// RUN: FileCheck --input-file=%t-aarch64.cir %s --check-prefix=AARCH64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o %t-ogcg.ll %s
// RUN: FileCheck --input-file=%t-ogcg.ll %s --check-prefix=OGCG

// lround

long my_lroundf(float f) {
  return __builtin_lroundf(f);
  // CIR: cir.func no_inline dso_local @my_lroundf
  // CIR: cir.lround %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @my_lroundf
  // LLVM:   call i64 @llvm.lround.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_lroundf(
  // OGCG: call i64 @llvm.lround.i64.f32(
}

long my_lround(double f) {
  return __builtin_lround(f);
  // CIR: cir.func no_inline dso_local @my_lround
  // CIR: cir.lround %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @my_lround
  // LLVM:   call i64 @llvm.lround.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_lround(
  // OGCG: call i64 @llvm.lround.i64.f64(
}

long my_lroundl(long double f) {
  return __builtin_lroundl(f);
  // CIR: cir.func no_inline dso_local @my_lroundl
  // CIR: cir.lround %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.lround %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @my_lroundl
  // LLVM:   call i64 @llvm.lround.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_lroundl(
  // OGCG: call i64 @llvm.lround.i64.f80(
}

long lroundf(float);
long lround(double);
long lroundl(long double);

long call_lroundf(float f) {
  return lroundf(f);
  // CIR: cir.func no_inline dso_local @call_lroundf
  // CIR: cir.lround %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @call_lroundf
  // LLVM:   call i64 @llvm.lround.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_lroundf(
  // OGCG: call i64 @llvm.lround.i64.f32(
}

long call_lround(double f) {
  return lround(f);
  // CIR: cir.func no_inline dso_local @call_lround
  // CIR: cir.lround %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @call_lround
  // LLVM:   call i64 @llvm.lround.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_lround(
  // OGCG: call i64 @llvm.lround.i64.f64(
}

long call_lroundl(long double f) {
  return lroundl(f);
  // CIR: cir.func no_inline dso_local @call_lroundl
  // CIR: cir.lround %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.lround %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @call_lroundl
  // LLVM:   call i64 @llvm.lround.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_lroundl(
  // OGCG: call i64 @llvm.lround.i64.f80(
}

// llround

long long my_llroundf(float f) {
  return __builtin_llroundf(f);
  // CIR: cir.func no_inline dso_local @my_llroundf
  // CIR: cir.llround %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @my_llroundf
  // LLVM:   call i64 @llvm.llround.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_llroundf(
  // OGCG: call i64 @llvm.llround.i64.f32(
}

long long my_llround(double f) {
  return __builtin_llround(f);
  // CIR: cir.func no_inline dso_local @my_llround
  // CIR: cir.llround %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @my_llround
  // LLVM:   call i64 @llvm.llround.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_llround(
  // OGCG: call i64 @llvm.llround.i64.f64(
}

long long my_llroundl(long double f) {
  return __builtin_llroundl(f);
  // CIR: cir.func no_inline dso_local @my_llroundl
  // CIR: cir.llround %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.llround %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @my_llroundl
  // LLVM:   call i64 @llvm.llround.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_llroundl(
  // OGCG: call i64 @llvm.llround.i64.f80(
}

long long llroundf(float);
long long llround(double);
long long llroundl(long double);

long long call_llroundf(float f) {
  return llroundf(f);
  // CIR: cir.func no_inline dso_local @call_llroundf
  // CIR: cir.llround %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @call_llroundf
  // LLVM:   call i64 @llvm.llround.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_llroundf(
  // OGCG: call i64 @llvm.llround.i64.f32(
}

long long call_llround(double f) {
  return llround(f);
  // CIR: cir.func no_inline dso_local @call_llround
  // CIR: cir.llround %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @call_llround
  // LLVM:   call i64 @llvm.llround.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_llround(
  // OGCG: call i64 @llvm.llround.i64.f64(
}

long long call_llroundl(long double f) {
  return llroundl(f);
  // CIR: cir.func no_inline dso_local @call_llroundl
  // CIR: cir.llround %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.llround %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @call_llroundl
  // LLVM:   call i64 @llvm.llround.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_llroundl(
  // OGCG: call i64 @llvm.llround.i64.f80(
}

// lrint

long my_lrintf(float f) {
  return __builtin_lrintf(f);
  // CIR: cir.func no_inline dso_local @my_lrintf
  // CIR: cir.lrint %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @my_lrintf
  // LLVM:   call i64 @llvm.lrint.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_lrintf(
  // OGCG: call i64 @llvm.lrint.i64.f32(
}

long my_lrint(double f) {
  return __builtin_lrint(f);
  // CIR: cir.func no_inline dso_local @my_lrint
  // CIR: cir.lrint %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @my_lrint
  // LLVM:   call i64 @llvm.lrint.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_lrint(
  // OGCG: call i64 @llvm.lrint.i64.f64(
}

long my_lrintl(long double f) {
  return __builtin_lrintl(f);
  // CIR: cir.func no_inline dso_local @my_lrintl
  // CIR: cir.lrint %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.lrint %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @my_lrintl
  // LLVM:   call i64 @llvm.lrint.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_lrintl(
  // OGCG: call i64 @llvm.lrint.i64.f80(
}

long lrintf(float);
long lrint(double);
long lrintl(long double);

long call_lrintf(float f) {
  return lrintf(f);
  // CIR: cir.func no_inline dso_local @call_lrintf
  // CIR: cir.lrint %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @call_lrintf
  // LLVM:   call i64 @llvm.lrint.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_lrintf(
  // OGCG: call i64 @llvm.lrint.i64.f32(
}

long call_lrint(double f) {
  return lrint(f);
  // CIR: cir.func no_inline dso_local @call_lrint
  // CIR: cir.lrint %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @call_lrint
  // LLVM:   call i64 @llvm.lrint.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_lrint(
  // OGCG: call i64 @llvm.lrint.i64.f64(
}

long call_lrintl(long double f) {
  return lrintl(f);
  // CIR: cir.func no_inline dso_local @call_lrintl
  // CIR: cir.lrint %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.lrint %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @call_lrintl
  // LLVM:   call i64 @llvm.lrint.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_lrintl(
  // OGCG: call i64 @llvm.lrint.i64.f80(
}

// llrint

long long my_llrintf(float f) {
  return __builtin_llrintf(f);
  // CIR: cir.func no_inline dso_local @my_llrintf
  // CIR: cir.llrint %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @my_llrintf
  // LLVM:   call i64 @llvm.llrint.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_llrintf(
  // OGCG: call i64 @llvm.llrint.i64.f32(
}

long long my_llrint(double f) {
  return __builtin_llrint(f);
  // CIR: cir.func no_inline dso_local @my_llrint
  // CIR: cir.llrint %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @my_llrint
  // LLVM:   call i64 @llvm.llrint.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_llrint(
  // OGCG: call i64 @llvm.llrint.i64.f64(
}

long long my_llrintl(long double f) {
  return __builtin_llrintl(f);
  // CIR: cir.func no_inline dso_local @my_llrintl
  // CIR: cir.llrint %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.llrint %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @my_llrintl
  // LLVM:   call i64 @llvm.llrint.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_llrintl(
  // OGCG: call i64 @llvm.llrint.i64.f80(
}

long long llrintf(float);
long long llrint(double);
long long llrintl(long double);

long long call_llrintf(float f) {
  return llrintf(f);
  // CIR: cir.func no_inline dso_local @call_llrintf
  // CIR: cir.llrint %{{.+}} : !cir.float -> !s64i

  // LLVM: define dso_local i64 @call_llrintf
  // LLVM:   call i64 @llvm.llrint.i64.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_llrintf(
  // OGCG: call i64 @llvm.llrint.i64.f32(
}

long long call_llrint(double f) {
  return llrint(f);
  // CIR: cir.func no_inline dso_local @call_llrint
  // CIR: cir.llrint %{{.+}} : !cir.double -> !s64i

  // LLVM: define dso_local i64 @call_llrint
  // LLVM:   call i64 @llvm.llrint.i64.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_llrint(
  // OGCG: call i64 @llvm.llrint.i64.f64(
}

long long call_llrintl(long double f) {
  return llrintl(f);
  // CIR: cir.func no_inline dso_local @call_llrintl
  // CIR: cir.llrint %{{.+}} : !cir.long_double<!cir.f80> -> !s64i
  // AARCH64: cir.llrint %{{.+}} : !cir.long_double<!cir.double> -> !s64i

  // LLVM: define dso_local i64 @call_llrintl
  // LLVM:   call i64 @llvm.llrint.i64.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_llrintl(
  // OGCG: call i64 @llvm.llrint.i64.f80(
}

// ceil

float my_ceilf(float f) {
  return __builtin_ceilf(f);
  // CIR: cir.func no_inline dso_local @my_ceilf
  // CIR: {{.+}} = cir.ceil {{.+}} : !cir.float

  // LLVM: define dso_local float @my_ceilf(float noundef %0)
  // LLVM:   call float @llvm.ceil.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_ceilf(
  // OGCG: call float @llvm.ceil.f32(
}

double my_ceil(double f) {
  return __builtin_ceil(f);
  // CIR: cir.func no_inline dso_local @my_ceil
  // CIR: {{.+}} = cir.ceil {{.+}} : !cir.double

  // LLVM: define dso_local double @my_ceil(double noundef %0)
  // LLVM:   call double @llvm.ceil.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_ceil(
  // OGCG: call double @llvm.ceil.f64(
}

long double my_ceill(long double f) {
  return __builtin_ceill(f);
  // CIR: cir.func no_inline dso_local @my_ceill
  // CIR: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_ceill(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.ceil.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_ceill(
  // OGCG: call x86_fp80 @llvm.ceil.f80(
}

float ceilf(float);
double ceil(double);
long double ceill(long double);

float call_ceilf(float f) {
  return ceilf(f);
  // CIR: cir.func no_inline dso_local @call_ceilf
  // CIR: {{.+}} = cir.ceil {{.+}} : !cir.float

  // LLVM: define dso_local float @call_ceilf(float noundef %0)
  // LLVM:   call float @llvm.ceil.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_ceilf(
  // OGCG: call float @llvm.ceil.f32(
}

double call_ceil(double f) {
  return ceil(f);
  // CIR: cir.func no_inline dso_local @call_ceil
  // CIR: {{.+}} = cir.ceil {{.+}} : !cir.double

  // LLVM: define dso_local double @call_ceil(double noundef %0)
  // LLVM:   call double @llvm.ceil.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_ceil(
  // OGCG: call double @llvm.ceil.f64(
}

long double call_ceill(long double f) {
  return ceill(f);
  // CIR: cir.func no_inline dso_local @call_ceill
  // CIR: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_ceill(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.ceil.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_ceill(
  // OGCG: call x86_fp80 @llvm.ceil.f80(
}

// cos

float my_cosf(float f) {
  return __builtin_cosf(f);
  // CIR: cir.func no_inline dso_local @my_cosf
  // CIR: {{.+}} = cir.cos {{.+}} : !cir.float

  // LLVM: define dso_local float @my_cosf(float noundef %0)
  // LLVM:   call float @llvm.cos.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_cosf(
  // OGCG: call float @llvm.cos.f32(
}

double my_cos(double f) {
  return __builtin_cos(f);
  // CIR: cir.func no_inline dso_local @my_cos
  // CIR: {{.+}} = cir.cos {{.+}} : !cir.double

  // LLVM: define dso_local double @my_cos(double noundef %0)
  // LLVM:   call double @llvm.cos.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_cos(
  // OGCG: call double @llvm.cos.f64(
}

long double my_cosl(long double f) {
  return __builtin_cosl(f);
  // CIR: cir.func no_inline dso_local @my_cosl
  // CIR: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_cosl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.cos.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_cosl(
  // OGCG: call x86_fp80 @llvm.cos.f80(
}

float cosf(float);
double cos(double);
long double cosl(long double);

float call_cosf(float f) {
  return cosf(f);
  // CIR: cir.func no_inline dso_local @call_cosf
  // CIR: {{.+}} = cir.cos {{.+}} : !cir.float

  // LLVM: define dso_local float @call_cosf(float noundef %0)
  // LLVM:   call float @llvm.cos.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_cosf(
  // OGCG: call float @llvm.cos.f32(
}

double call_cos(double f) {
  return cos(f);
  // CIR: cir.func no_inline dso_local @call_cos
  // CIR: {{.+}} = cir.cos {{.+}} : !cir.double

  // LLVM: define dso_local double @call_cos(double noundef %0)
  // LLVM:   call double @llvm.cos.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_cos(
  // OGCG: call double @llvm.cos.f64(
}

long double call_cosl(long double f) {
  return cosl(f);
  // CIR: cir.func no_inline dso_local @call_cosl
  // CIR: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_cosl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.cos.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_cosl(
  // OGCG: call x86_fp80 @llvm.cos.f80(
}

// exp

float my_expf(float f) {
  return __builtin_expf(f);
  // CIR: cir.func no_inline dso_local @my_expf
  // CIR: {{.+}} = cir.exp {{.+}} : !cir.float

  // LLVM: define dso_local float @my_expf(float noundef %0)
  // LLVM:   call float @llvm.exp.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_expf(
  // OGCG: call float @llvm.exp.f32(
}

double my_exp(double f) {
  return __builtin_exp(f);
  // CIR: cir.func no_inline dso_local @my_exp
  // CIR: {{.+}} = cir.exp {{.+}} : !cir.double

  // LLVM: define dso_local double @my_exp(double noundef %0)
  // LLVM:   call double @llvm.exp.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_exp(
  // OGCG: call double @llvm.exp.f64(
}

long double my_expl(long double f) {
  return __builtin_expl(f);
  // CIR: cir.func no_inline dso_local @my_expl
  // CIR: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_expl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.exp.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_expl(
  // OGCG: call x86_fp80 @llvm.exp.f80(
}

float expf(float);
double exp(double);
long double expl(long double);

float call_expf(float f) {
  return expf(f);
  // CIR: cir.func no_inline dso_local @call_expf
  // CIR: {{.+}} = cir.exp {{.+}} : !cir.float

  // LLVM: define dso_local float @call_expf(float noundef %0)
  // LLVM:   call float @llvm.exp.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_expf(
  // OGCG: call float @llvm.exp.f32(
}

double call_exp(double f) {
  return exp(f);
  // CIR: cir.func no_inline dso_local @call_exp
  // CIR: {{.+}} = cir.exp {{.+}} : !cir.double

  // LLVM: define dso_local double @call_exp(double noundef %0)
  // LLVM:   call double @llvm.exp.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_exp(
  // OGCG: call double @llvm.exp.f64(
}

long double call_expl(long double f) {
  return expl(f);
  // CIR: cir.func no_inline dso_local @call_expl
  // CIR: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_expl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.exp.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_expl(
  // OGCG: call x86_fp80 @llvm.exp.f80(
}

// exp2

float my_exp2f(float f) {
  return __builtin_exp2f(f);
  // CIR: cir.func no_inline dso_local @my_exp2f
  // CIR: {{.+}} = cir.exp2 {{.+}} : !cir.float

  // LLVM: define dso_local float @my_exp2f(float noundef %0)
  // LLVM:   call float @llvm.exp2.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_exp2f(
  // OGCG: call float @llvm.exp2.f32(
}

double my_exp2(double f) {
  return __builtin_exp2(f);
  // CIR: cir.func no_inline dso_local @my_exp2
  // CIR: {{.+}} = cir.exp2 {{.+}} : !cir.double

  // LLVM: define dso_local double @my_exp2(double noundef %0)
  // LLVM:   call double @llvm.exp2.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_exp2(
  // OGCG: call double @llvm.exp2.f64(
}

long double my_exp2l(long double f) {
  return __builtin_exp2l(f);
  // CIR: cir.func no_inline dso_local @my_exp2l
  // CIR: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_exp2l(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.exp2.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_exp2l(
  // OGCG: call x86_fp80 @llvm.exp2.f80(
}

float exp2f(float);
double exp2(double);
long double exp2l(long double);

float call_exp2f(float f) {
  return exp2f(f);
  // CIR: cir.func no_inline dso_local @call_exp2f
  // CIR: {{.+}} = cir.exp2 {{.+}} : !cir.float

  // LLVM: define dso_local float @call_exp2f(float noundef %0)
  // LLVM:   call float @llvm.exp2.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_exp2f(
  // OGCG: call float @llvm.exp2.f32(
}

double call_exp2(double f) {
  return exp2(f);
  // CIR: cir.func no_inline dso_local @call_exp2
  // CIR: {{.+}} = cir.exp2 {{.+}} : !cir.double

  // LLVM: define dso_local double @call_exp2(double noundef %0)
  // LLVM:   call double @llvm.exp2.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_exp2(
  // OGCG: call double @llvm.exp2.f64(
}

long double call_exp2l(long double f) {
  return exp2l(f);
  // CIR: cir.func no_inline dso_local @call_exp2l
  // CIR: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_exp2l(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.exp2.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_exp2l(
  // OGCG: call x86_fp80 @llvm.exp2.f80(
}

// floor

float my_floorf(float f) {
  return __builtin_floorf(f);
  // CIR: cir.func no_inline dso_local @my_floorf
  // CIR: {{.+}} = cir.floor {{.+}} : !cir.float

  // LLVM: define dso_local float @my_floorf(float noundef %0)
  // LLVM:   call float @llvm.floor.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_floorf(
  // OGCG: call float @llvm.floor.f32(
}

double my_floor(double f) {
  return __builtin_floor(f);
  // CIR: cir.func no_inline dso_local @my_floor
  // CIR: {{.+}} = cir.floor {{.+}} : !cir.double

  // LLVM: define dso_local double @my_floor(double noundef %0)
  // LLVM:   call double @llvm.floor.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_floor(
  // OGCG: call double @llvm.floor.f64(
}

long double my_floorl(long double f) {
  return __builtin_floorl(f);
  // CIR: cir.func no_inline dso_local @my_floorl
  // CIR: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_floorl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.floor.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_floorl(
  // OGCG: call x86_fp80 @llvm.floor.f80(
}

float floorf(float);
double floor(double);
long double floorl(long double);

float call_floorf(float f) {
  return floorf(f);
  // CIR: cir.func no_inline dso_local @call_floorf
  // CIR: {{.+}} = cir.floor {{.+}} : !cir.float

  // LLVM: define dso_local float @call_floorf(float noundef %0)
  // LLVM:   call float @llvm.floor.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_floorf(
  // OGCG: call float @llvm.floor.f32(
}

double call_floor(double f) {
  return floor(f);
  // CIR: cir.func no_inline dso_local @call_floor
  // CIR: {{.+}} = cir.floor {{.+}} : !cir.double

  // LLVM: define dso_local double @call_floor(double noundef %0)
  // LLVM:   call double @llvm.floor.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_floor(
  // OGCG: call double @llvm.floor.f64(
}

long double call_floorl(long double f) {
  return floorl(f);
  // CIR: cir.func no_inline dso_local @call_floorl
  // CIR: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_floorl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.floor.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_floorl(
  // OGCG: call x86_fp80 @llvm.floor.f80(
}

// log

float my_logf(float f) {
  return __builtin_logf(f);
  // CIR: cir.func no_inline dso_local @my_logf
  // CIR: {{.+}} = cir.log {{.+}} : !cir.float

  // LLVM: define dso_local float @my_logf(float noundef %0)
  // LLVM:   call float @llvm.log.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_logf(
  // OGCG: call float @llvm.log.f32(
}

double my_log(double f) {
  return __builtin_log(f);
  // CIR: cir.func no_inline dso_local @my_log
  // CIR: {{.+}} = cir.log {{.+}} : !cir.double

  // LLVM: define dso_local double @my_log(double noundef %0)
  // LLVM:   call double @llvm.log.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_log(
  // OGCG: call double @llvm.log.f64(
}

long double my_logl(long double f) {
  return __builtin_logl(f);
  // CIR: cir.func no_inline dso_local @my_logl
  // CIR: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_logl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.log.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_logl(
  // OGCG: call x86_fp80 @llvm.log.f80(
}

float logf(float);
double log(double);
long double logl(long double);

float call_logf(float f) {
  return logf(f);
  // CIR: cir.func no_inline dso_local @call_logf
  // CIR: {{.+}} = cir.log {{.+}} : !cir.float

  // LLVM: define dso_local float @call_logf(float noundef %0)
  // LLVM:   call float @llvm.log.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_logf(
  // OGCG: call float @llvm.log.f32(
}

double call_log(double f) {
  return log(f);
  // CIR: cir.func no_inline dso_local @call_log
  // CIR: {{.+}} = cir.log {{.+}} : !cir.double

  // LLVM: define dso_local double @call_log(double noundef %0)
  // LLVM:   call double @llvm.log.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_log(
  // OGCG: call double @llvm.log.f64(
}

long double call_logl(long double f) {
  return logl(f);
  // CIR: cir.func no_inline dso_local @call_logl
  // CIR: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_logl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.log.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_logl(
  // OGCG: call x86_fp80 @llvm.log.f80(
}

// log10

float my_log10f(float f) {
  return __builtin_log10f(f);
  // CIR: cir.func no_inline dso_local @my_log10f
  // CIR: {{.+}} = cir.log10 {{.+}} : !cir.float

  // LLVM: define dso_local float @my_log10f(float noundef %0)
  // LLVM:   call float @llvm.log10.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_log10f(
  // OGCG: call float @llvm.log10.f32(
}

double my_log10(double f) {
  return __builtin_log10(f);
  // CIR: cir.func no_inline dso_local @my_log10
  // CIR: {{.+}} = cir.log10 {{.+}} : !cir.double

  // LLVM: define dso_local double @my_log10(double noundef %0)
  // LLVM:   call double @llvm.log10.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_log10(
  // OGCG: call double @llvm.log10.f64(
}

long double my_log10l(long double f) {
  return __builtin_log10l(f);
  // CIR: cir.func no_inline dso_local @my_log10l
  // CIR: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_log10l(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.log10.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_log10l(
  // OGCG: call x86_fp80 @llvm.log10.f80(
}

float log10f(float);
double log10(double);
long double log10l(long double);

float call_log10f(float f) {
  return log10f(f);
  // CIR: cir.func no_inline dso_local @call_log10f
  // CIR: {{.+}} = cir.log10 {{.+}} : !cir.float

  // LLVM: define dso_local float @call_log10f(float noundef %0)
  // LLVM:   call float @llvm.log10.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_log10f(
  // OGCG: call float @llvm.log10.f32(
}

double call_log10(double f) {
  return log10(f);
  // CIR: cir.func no_inline dso_local @call_log10
  // CIR: {{.+}} = cir.log10 {{.+}} : !cir.double

  // LLVM: define dso_local double @call_log10(double noundef %0)
  // LLVM:   call double @llvm.log10.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_log10(
  // OGCG: call double @llvm.log10.f64(
}

long double call_log10l(long double f) {
  return log10l(f);
  // CIR: cir.func no_inline dso_local @call_log10l
  // CIR: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_log10l(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.log10.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_log10l(
  // OGCG: call x86_fp80 @llvm.log10.f80(
}

// log2

float my_log2f(float f) {
  return __builtin_log2f(f);
  // CIR: cir.func no_inline dso_local @my_log2f
  // CIR: {{.+}} = cir.log2 {{.+}} : !cir.float

  // LLVM: define dso_local float @my_log2f(float noundef %0)
  // LLVM:   call float @llvm.log2.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_log2f(
  // OGCG: call float @llvm.log2.f32(
}

double my_log2(double f) {
  return __builtin_log2(f);
  // CIR: cir.func no_inline dso_local @my_log2
  // CIR: {{.+}} = cir.log2 {{.+}} : !cir.double

  // LLVM: define dso_local double @my_log2(double noundef %0)
  // LLVM:   call double @llvm.log2.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_log2(
  // OGCG: call double @llvm.log2.f64(
}

long double my_log2l(long double f) {
  return __builtin_log2l(f);
  // CIR: cir.func no_inline dso_local @my_log2l
  // CIR: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_log2l(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.log2.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_log2l(
  // OGCG: call x86_fp80 @llvm.log2.f80(
}

float log2f(float);
double log2(double);
long double log2l(long double);

float call_log2f(float f) {
  return log2f(f);
  // CIR: cir.func no_inline dso_local @call_log2f
  // CIR: {{.+}} = cir.log2 {{.+}} : !cir.float

  // LLVM: define dso_local float @call_log2f(float noundef %0)
  // LLVM:   call float @llvm.log2.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_log2f(
  // OGCG: call float @llvm.log2.f32(
}

double call_log2(double f) {
  return log2(f);
  // CIR: cir.func no_inline dso_local @call_log2
  // CIR: {{.+}} = cir.log2 {{.+}} : !cir.double

  // LLVM: define dso_local double @call_log2(double noundef %0)
  // LLVM:   call double @llvm.log2.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_log2(
  // OGCG: call double @llvm.log2.f64(
}

long double call_log2l(long double f) {
  return log2l(f);
  // CIR: cir.func no_inline dso_local @call_log2l
  // CIR: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_log2l(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.log2.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_log2l(
  // OGCG: call x86_fp80 @llvm.log2.f80(
}

// nearbyint

float my_nearbyintf(float f) {
  return __builtin_nearbyintf(f);
  // CIR: cir.func no_inline dso_local @my_nearbyintf
  // CIR: {{.+}} = cir.nearbyint {{.+}} : !cir.float

  // LLVM: define dso_local float @my_nearbyintf(float noundef %0)
  // LLVM:   call float @llvm.nearbyint.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_nearbyintf(
  // OGCG: call float @llvm.nearbyint.f32(
}

double my_nearbyint(double f) {
  return __builtin_nearbyint(f);
  // CIR: cir.func no_inline dso_local @my_nearbyint
  // CIR: {{.+}} = cir.nearbyint {{.+}} : !cir.double

  // LLVM: define dso_local double @my_nearbyint(double noundef %0)
  // LLVM:   call double @llvm.nearbyint.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_nearbyint(
  // OGCG: call double @llvm.nearbyint.f64(
}

long double my_nearbyintl(long double f) {
  return __builtin_nearbyintl(f);
  // CIR: cir.func no_inline dso_local @my_nearbyintl
  // CIR: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_nearbyintl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.nearbyint.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_nearbyintl(
  // OGCG: call x86_fp80 @llvm.nearbyint.f80(
}

float nearbyintf(float);
double nearbyint(double);
long double nearbyintl(long double);

float call_nearbyintf(float f) {
  return nearbyintf(f);
  // CIR: cir.func no_inline dso_local @call_nearbyintf
  // CIR: {{.+}} = cir.nearbyint {{.+}} : !cir.float

  // LLVM: define dso_local float @call_nearbyintf(float noundef %0)
  // LLVM:   call float @llvm.nearbyint.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_nearbyintf(
  // OGCG: call float @llvm.nearbyint.f32(
}

double call_nearbyint(double f) {
  return nearbyint(f);
  // CIR: cir.func no_inline dso_local @call_nearbyint
  // CIR: {{.+}} = cir.nearbyint {{.+}} : !cir.double

  // LLVM: define dso_local double @call_nearbyint(double noundef %0)
  // LLVM:   call double @llvm.nearbyint.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_nearbyint(
  // OGCG: call double @llvm.nearbyint.f64(
}

long double call_nearbyintl(long double f) {
  return nearbyintl(f);
  // CIR: cir.func no_inline dso_local @call_nearbyintl
  // CIR: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_nearbyintl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.nearbyint.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_nearbyintl(
  // OGCG: call x86_fp80 @llvm.nearbyint.f80(
}

// rint

float my_rintf(float f) {
  return __builtin_rintf(f);
  // CIR: cir.func no_inline dso_local @my_rintf
  // CIR: {{.+}} = cir.rint {{.+}} : !cir.float

  // LLVM: define dso_local float @my_rintf(float noundef %0)
  // LLVM:   call float @llvm.rint.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_rintf(
  // OGCG: call float @llvm.rint.f32(
}

double my_rint(double f) {
  return __builtin_rint(f);
  // CIR: cir.func no_inline dso_local @my_rint
  // CIR: {{.+}} = cir.rint {{.+}} : !cir.double

  // LLVM: define dso_local double @my_rint(double noundef %0)
  // LLVM:   call double @llvm.rint.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_rint(
  // OGCG: call double @llvm.rint.f64(
}

long double my_rintl(long double f) {
  return __builtin_rintl(f);
  // CIR: cir.func no_inline dso_local @my_rintl
  // CIR: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_rintl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.rint.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_rintl(
  // OGCG: call x86_fp80 @llvm.rint.f80(
}

float rintf(float);
double rint(double);
long double rintl(long double);

float call_rintf(float f) {
  return rintf(f);
  // CIR: cir.func no_inline dso_local @call_rintf
  // CIR: {{.+}} = cir.rint {{.+}} : !cir.float

  // LLVM: define dso_local float @call_rintf(float noundef %0)
  // LLVM:   call float @llvm.rint.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_rintf(
  // OGCG: call float @llvm.rint.f32(
}

double call_rint(double f) {
  return rint(f);
  // CIR: cir.func no_inline dso_local @call_rint
  // CIR: {{.+}} = cir.rint {{.+}} : !cir.double

  // LLVM: define dso_local double @call_rint(double noundef %0)
  // LLVM:   call double @llvm.rint.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_rint(
  // OGCG: call double @llvm.rint.f64(
}

long double call_rintl(long double f) {
  return rintl(f);
  // CIR: cir.func no_inline dso_local @call_rintl
  // CIR: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_rintl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.rint.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_rintl(
  // OGCG: call x86_fp80 @llvm.rint.f80(
}

// round

float my_roundf(float f) {
  return __builtin_roundf(f);
  // CIR: cir.func no_inline dso_local @my_roundf
  // CIR: {{.+}} = cir.round {{.+}} : !cir.float

  // LLVM: define dso_local float @my_roundf(float noundef %0)
  // LLVM:   call float @llvm.round.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_roundf(
  // OGCG: call float @llvm.round.f32(
}

double my_round(double f) {
  return __builtin_round(f);
  // CIR: cir.func no_inline dso_local @my_round
  // CIR: {{.+}} = cir.round {{.+}} : !cir.double

  // LLVM: define dso_local double @my_round(double noundef %0)
  // LLVM:   call double @llvm.round.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_round(
  // OGCG: call double @llvm.round.f64(
}

long double my_roundl(long double f) {
  return __builtin_roundl(f);
  // CIR: cir.func no_inline dso_local @my_roundl
  // CIR: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_roundl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.round.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_roundl(
  // OGCG: call x86_fp80 @llvm.round.f80(
}

float roundf(float);
double round(double);
long double roundl(long double);

float call_roundf(float f) {
  return roundf(f);
  // CIR: cir.func no_inline dso_local @call_roundf
  // CIR: {{.+}} = cir.round {{.+}} : !cir.float

  // LLVM: define dso_local float @call_roundf(float noundef %0)
  // LLVM:   call float @llvm.round.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_roundf(
  // OGCG: call float @llvm.round.f32(
}

double call_round(double f) {
  return round(f);
  // CIR: cir.func no_inline dso_local @call_round
  // CIR: {{.+}} = cir.round {{.+}} : !cir.double

  // LLVM: define dso_local double @call_round(double noundef %0)
  // LLVM:   call double @llvm.round.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_round(
  // OGCG: call double @llvm.round.f64(
}

long double call_roundl(long double f) {
  return roundl(f);
  // CIR: cir.func no_inline dso_local @call_roundl
  // CIR: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_roundl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.round.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_roundl(
  // OGCG: call x86_fp80 @llvm.round.f80(
}

// sin

float my_sinf(float f) {
  return __builtin_sinf(f);
  // CIR: cir.func no_inline dso_local @my_sinf
  // CIR: {{.+}} = cir.sin {{.+}} : !cir.float

  // LLVM: define dso_local float @my_sinf(float noundef %0)
  // LLVM:   call float @llvm.sin.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_sinf(
  // OGCG: call float @llvm.sin.f32(
}

double my_sin(double f) {
  return __builtin_sin(f);
  // CIR: cir.func no_inline dso_local @my_sin
  // CIR: {{.+}} = cir.sin {{.+}} : !cir.double

  // LLVM: define dso_local double @my_sin(double noundef %0)
  // LLVM:   call double @llvm.sin.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_sin(
  // OGCG: call double @llvm.sin.f64(
}

long double my_sinl(long double f) {
  return __builtin_sinl(f);
  // CIR: cir.func no_inline dso_local @my_sinl
  // CIR: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_sinl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.sin.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_sinl(
  // OGCG: call x86_fp80 @llvm.sin.f80(
}

float sinf(float);
double sin(double);
long double sinl(long double);

float call_sinf(float f) {
  return sinf(f);
  // CIR: cir.func no_inline dso_local @call_sinf
  // CIR: {{.+}} = cir.sin {{.+}} : !cir.float

  // LLVM: define dso_local float @call_sinf(float noundef %0)
  // LLVM:   call float @llvm.sin.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_sinf(
  // OGCG: call float @llvm.sin.f32(
}

double call_sin(double f) {
  return sin(f);
  // CIR: cir.func no_inline dso_local @call_sin
  // CIR: {{.+}} = cir.sin {{.+}} : !cir.double

  // LLVM: define dso_local double @call_sin(double noundef %0)
  // LLVM:   call double @llvm.sin.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_sin(
  // OGCG: call double @llvm.sin.f64(
}

long double call_sinl(long double f) {
  return sinl(f);
  // CIR: cir.func no_inline dso_local @call_sinl
  // CIR: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_sinl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.sin.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_sinl(
  // OGCG: call x86_fp80 @llvm.sin.f80(
}

// sqrt

float my_sqrtf(float f) {
  return __builtin_sqrtf(f);
  // CIR: cir.func no_inline dso_local @my_sqrtf
  // CIR: {{.+}} = cir.sqrt {{.+}} : !cir.float

  // LLVM: define dso_local float @my_sqrtf(float noundef %0)
  // LLVM:   call float @llvm.sqrt.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_sqrtf(
  // OGCG: call float @llvm.sqrt.f32(
}

double my_sqrt(double f) {
  return __builtin_sqrt(f);
  // CIR: cir.func no_inline dso_local @my_sqrt
  // CIR: {{.+}} = cir.sqrt {{.+}} : !cir.double

  // LLVM: define dso_local double @my_sqrt(double noundef %0)
  // LLVM:   call double @llvm.sqrt.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_sqrt(
  // OGCG: call double @llvm.sqrt.f64(
}

long double my_sqrtl(long double f) {
  return __builtin_sqrtl(f);
  // CIR: cir.func no_inline dso_local @my_sqrtl
  // CIR: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_sqrtl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.sqrt.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_sqrtl(
  // OGCG: call x86_fp80 @llvm.sqrt.f80(
}

float sqrtf(float);
double sqrt(double);
long double sqrtl(long double);

float call_sqrtf(float f) {
  return sqrtf(f);
  // CIR: cir.func no_inline dso_local @call_sqrtf
  // CIR: {{.+}} = cir.sqrt {{.+}} : !cir.float

  // LLVM: define dso_local float @call_sqrtf(float noundef %0)
  // LLVM:   call float @llvm.sqrt.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_sqrtf(
  // OGCG: call float @llvm.sqrt.f32(
}

double call_sqrt(double f) {
  return sqrt(f);
  // CIR: cir.func no_inline dso_local @call_sqrt
  // CIR: {{.+}} = cir.sqrt {{.+}} : !cir.double

  // LLVM: define dso_local double @call_sqrt(double noundef %0)
  // LLVM:   call double @llvm.sqrt.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_sqrt(
  // OGCG: call double @llvm.sqrt.f64(
}

long double call_sqrtl(long double f) {
  return sqrtl(f);
  // CIR: cir.func no_inline dso_local @call_sqrtl
  // CIR: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_sqrtl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.sqrt.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_sqrtl(
  // OGCG: call x86_fp80 @llvm.sqrt.f80(
}

// tan

float my_tanf(float f) {
  return __builtin_tanf(f);
  // CIR: cir.func no_inline dso_local @my_tanf
  // CIR: {{.+}} = cir.tan {{.+}} : !cir.float

  // LLVM: define dso_local float @my_tanf(float noundef %0)
  // LLVM:   call float @llvm.tan.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_tanf(
  // OGCG: call float @llvm.tan.f32(
}

double my_tan(double f) {
  return __builtin_tan(f);
  // CIR: cir.func no_inline dso_local @my_tan
  // CIR: {{.+}} = cir.tan {{.+}} : !cir.double

  // LLVM: define dso_local double @my_tan(double noundef %0)
  // LLVM:   call double @llvm.tan.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_tan(
  // OGCG: call double @llvm.tan.f64(
}

long double my_tanl(long double f) {
  return __builtin_tanl(f);
  // CIR: cir.func no_inline dso_local @my_tanl
  // CIR: {{.+}} = cir.tan {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.tan {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_tanl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.tan.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_tanl(
  // OGCG: call x86_fp80 @llvm.tan.f80(
}

float tanf(float);
double tan(double);
long double tanl(long double);

float call_tanf(float f) {
  return tanf(f);
  // CIR: cir.func no_inline dso_local @call_tanf
  // CIR: {{.+}} = cir.tan {{.+}} : !cir.float

  // LLVM: define dso_local float @call_tanf(float noundef %0)
  // LLVM:   call float @llvm.tan.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_tanf(
  // OGCG: call float @llvm.tan.f32(
}

double call_tan(double f) {
  return tan(f);
  // CIR: cir.func no_inline dso_local @call_tan
  // CIR: {{.+}} = cir.tan {{.+}} : !cir.double

  // LLVM: define dso_local double @call_tan(double noundef %0)
  // LLVM:   call double @llvm.tan.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_tan(
  // OGCG: call double @llvm.tan.f64(
}

long double call_tanl(long double f) {
  return tanl(f);
  // CIR: cir.func no_inline dso_local @call_tanl
  // CIR: {{.+}} = cir.tan {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.tan {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_tanl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.tan.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_tanl(
  // OGCG: call x86_fp80 @llvm.tan.f80(
}

// trunc

float my_truncf(float f) {
  return __builtin_truncf(f);
  // CIR: cir.func no_inline dso_local @my_truncf
  // CIR: {{.+}} = cir.trunc {{.+}} : !cir.float

  // LLVM: define dso_local float @my_truncf(float noundef %0)
  // LLVM:   call float @llvm.trunc.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_truncf(
  // OGCG: call float @llvm.trunc.f32(
}

double my_trunc(double f) {
  return __builtin_trunc(f);
  // CIR: cir.func no_inline dso_local @my_trunc
  // CIR: {{.+}} = cir.trunc {{.+}} : !cir.double

  // LLVM: define dso_local double @my_trunc(double noundef %0)
  // LLVM:   call double @llvm.trunc.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_trunc(
  // OGCG: call double @llvm.trunc.f64(
}

long double my_truncl(long double f) {
  return __builtin_truncl(f);
  // CIR: cir.func no_inline dso_local @my_truncl
  // CIR: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_truncl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.trunc.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_truncl(
  // OGCG: call x86_fp80 @llvm.trunc.f80(
}

float truncf(float);
double trunc(double);
long double truncl(long double);

float call_truncf(float f) {
  return truncf(f);
  // CIR: cir.func no_inline dso_local @call_truncf
  // CIR: {{.+}} = cir.trunc {{.+}} : !cir.float

  // LLVM: define dso_local float @call_truncf(float noundef %0)
  // LLVM:   call float @llvm.trunc.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_truncf(
  // OGCG: call float @llvm.trunc.f32(
}

double call_trunc(double f) {
  return trunc(f);
  // CIR: cir.func no_inline dso_local @call_trunc
  // CIR: {{.+}} = cir.trunc {{.+}} : !cir.double

  // LLVM: define dso_local double @call_trunc(double noundef %0)
  // LLVM:   call double @llvm.trunc.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_trunc(
  // OGCG: call double @llvm.trunc.f64(
}

long double call_truncl(long double f) {
  return truncl(f);
  // CIR: cir.func no_inline dso_local @call_truncl
  // CIR: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_truncl(x86_fp80 noundef %0)
  // LLVM:   call x86_fp80 @llvm.trunc.f80(x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_truncl(
  // OGCG: call x86_fp80 @llvm.trunc.f80(
}

// copysign

float my_copysignf(float x, float y) {
  return __builtin_copysignf(x, y);
  // CIR: cir.func no_inline dso_local @my_copysignf
  // CIR: cir.copysign %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_copysignf
  // LLVM:   call float @llvm.copysign.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_copysignf(
  // OGCG: call float @llvm.copysign.f32(
}

double my_copysign(double x, double y) {
  return __builtin_copysign(x, y);
  // CIR: cir.func no_inline dso_local @my_copysign
  // CIR: cir.copysign %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_copysign
  // LLVM:   call double @llvm.copysign.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_copysign(
  // OGCG: call double @llvm.copysign.f64(
}

long double my_copysignl(long double x, long double y) {
  return __builtin_copysignl(x, y);
  // CIR: cir.func no_inline dso_local @my_copysignl
  // CIR: cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_copysignl
  // LLVM:   call x86_fp80 @llvm.copysign.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_copysignl(
  // OGCG: call x86_fp80 @llvm.copysign.f80(
}

float copysignf(float, float);
double copysign(double, double);
long double copysignl(long double, long double);

float call_copysignf(float x, float y) {
  return copysignf(x, y);
  // CIR: cir.func no_inline dso_local @call_copysignf
  // CIR: cir.copysign %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @call_copysignf
  // LLVM:   call float @llvm.copysign.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_copysignf(
  // OGCG: call float @llvm.copysign.f32(
}

double call_copysign(double x, double y) {
  return copysign(x, y);
  // CIR: cir.func no_inline dso_local @call_copysign
  // CIR: cir.copysign %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @call_copysign
  // LLVM:   call double @llvm.copysign.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_copysign(
  // OGCG: call double @llvm.copysign.f64(
}

long double call_copysignl(long double x, long double y) {
  return copysignl(x, y);
  // CIR: cir.func no_inline dso_local @call_copysignl
  // CIR: cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_copysignl
  // LLVM:   call x86_fp80 @llvm.copysign.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_copysignl(
  // OGCG: call x86_fp80 @llvm.copysign.f80(
}

// fmax

float my_fmaxf(float x, float y) {
  return __builtin_fmaxf(x, y);
  // CIR: cir.func no_inline dso_local @my_fmaxf
  // CIR: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_fmaxf
  // LLVM:   call nsz float @llvm.maxnum.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_fmaxf(
  // OGCG: call nsz float @llvm.maxnum.f32(
}

double my_fmax(double x, double y) {
  return __builtin_fmax(x, y);
  // CIR: cir.func no_inline dso_local @my_fmax
  // CIR: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_fmax
  // LLVM:   call nsz double @llvm.maxnum.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_fmax(
  // OGCG: call nsz double @llvm.maxnum.f64(
}

long double my_fmaxl(long double x, long double y) {
  return __builtin_fmaxl(x, y);
  // CIR: cir.func no_inline dso_local @my_fmaxl
  // CIR: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_fmaxl
  // LLVM:   call nsz x86_fp80 @llvm.maxnum.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_fmaxl(
  // OGCG: call nsz x86_fp80 @llvm.maxnum.f80(
}

float fmaxf(float, float);
double fmax(double, double);
long double fmaxl(long double, long double);

float call_fmaxf(float x, float y) {
  return fmaxf(x, y);
  // CIR: cir.func no_inline dso_local @call_fmaxf
  // CIR: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @call_fmaxf
  // LLVM:   call nsz float @llvm.maxnum.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_fmaxf(
  // OGCG: call nsz float @llvm.maxnum.f32(
}

double call_fmax(double x, double y) {
  return fmax(x, y);
  // CIR: cir.func no_inline dso_local @call_fmax
  // CIR: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @call_fmax
  // LLVM:   call nsz double @llvm.maxnum.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_fmax(
  // OGCG: call nsz double @llvm.maxnum.f64(
}

long double call_fmaxl(long double x, long double y) {
  return fmaxl(x, y);
  // CIR: cir.func no_inline dso_local @call_fmaxl
  // CIR: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.fmaxnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_fmaxl
  // LLVM:   call nsz x86_fp80 @llvm.maxnum.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_fmaxl(
  // OGCG: call nsz x86_fp80 @llvm.maxnum.f80(
}

// fmin

float my_fminf(float x, float y) {
  return __builtin_fminf(x, y);
  // CIR: cir.func no_inline dso_local @my_fminf
  // CIR: cir.fminnum %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_fminf
  // LLVM:   call nsz float @llvm.minnum.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_fminf(
  // OGCG: call nsz float @llvm.minnum.f32(
}

double my_fmin(double x, double y) {
  return __builtin_fmin(x, y);
  // CIR: cir.func no_inline dso_local @my_fmin
  // CIR: cir.fminnum %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_fmin
  // LLVM:   call nsz double @llvm.minnum.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_fmin(
  // OGCG: call nsz double @llvm.minnum.f64(
}

long double my_fminl(long double x, long double y) {
  return __builtin_fminl(x, y);
  // CIR: cir.func no_inline dso_local @my_fminl
  // CIR: cir.fminnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.fminnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_fminl
  // LLVM:   call nsz x86_fp80 @llvm.minnum.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_fminl(
  // OGCG: call nsz x86_fp80 @llvm.minnum.f80(
}

float fminf(float, float);
double fmin(double, double);
long double fminl(long double, long double);

float call_fminf(float x, float y) {
  return fminf(x, y);
  // CIR: cir.func no_inline dso_local @call_fminf
  // CIR: cir.fminnum %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @call_fminf
  // LLVM:   call nsz float @llvm.minnum.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_fminf(
  // OGCG: call nsz float @llvm.minnum.f32(
}

double call_fmin(double x, double y) {
  return fmin(x, y);
  // CIR: cir.func no_inline dso_local @call_fmin
  // CIR: cir.fminnum %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @call_fmin
  // LLVM:   call nsz double @llvm.minnum.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_fmin(
  // OGCG: call nsz double @llvm.minnum.f64(
}

long double call_fminl(long double x, long double y) {
  return fminl(x, y);
  // CIR: cir.func no_inline dso_local @call_fminl
  // CIR: cir.fminnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.fminnum %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_fminl
  // LLVM:   call nsz x86_fp80 @llvm.minnum.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_fminl(
  // OGCG: call nsz x86_fp80 @llvm.minnum.f80(
}

// fmod

float my_fmodf(float x, float y) {
  return __builtin_fmodf(x, y);
  // CIR: cir.func no_inline dso_local @my_fmodf
  // CIR: cir.fmod %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_fmodf
  // LLVM:   frem float %{{.+}}, %{{.+}}
  // LLVM: }

  // OGCG: define{{.*}}@my_fmodf(
  // OGCG: frem float
}

double my_fmod(double x, double y) {
  return __builtin_fmod(x, y);
  // CIR: cir.func no_inline dso_local @my_fmod
  // CIR: cir.fmod %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_fmod
  // LLVM:   frem double %{{.+}}, %{{.+}}
  // LLVM: }

  // OGCG: define{{.*}}@my_fmod(
  // OGCG: frem double
}

long double my_fmodl(long double x, long double y) {
  return __builtin_fmodl(x, y);
  // CIR: cir.func no_inline dso_local @my_fmodl
  // CIR: cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_fmodl
  // LLVM:   frem x86_fp80 %{{.+}}, %{{.+}}
  // LLVM: }

  // OGCG: define{{.*}}@my_fmodl(
  // OGCG: frem x86_fp80
}

float fmodf(float, float);
double fmod(double, double);
long double fmodl(long double, long double);

float call_fmodf(float x, float y) {
  return fmodf(x, y);
  // CIR: cir.func no_inline dso_local @call_fmodf
  // CIR: cir.fmod %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @call_fmodf
  // LLVM:   frem float %{{.+}}, %{{.+}}
  // LLVM: }

  // OGCG: define{{.*}}@call_fmodf(
  // OGCG: frem float
}

double call_fmod(double x, double y) {
  return fmod(x, y);
  // CIR: cir.func no_inline dso_local @call_fmod
  // CIR: cir.fmod %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @call_fmod
  // LLVM:   frem double %{{.+}}, %{{.+}}
  // LLVM: }

  // OGCG: define{{.*}}@call_fmod(
  // OGCG: frem double
}

long double call_fmodl(long double x, long double y) {
  return fmodl(x, y);
  // CIR: cir.func no_inline dso_local @call_fmodl
  // CIR: cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_fmodl
  // LLVM:   frem x86_fp80 %{{.+}}, %{{.+}}
  // LLVM: }

  // OGCG: define{{.*}}@call_fmodl(
  // OGCG: frem x86_fp80
}

// pow

float my_powf(float x, float y) {
  return __builtin_powf(x, y);
  // CIR: cir.func no_inline dso_local @my_powf
  // CIR: cir.pow %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_powf
  // LLVM:   call float @llvm.pow.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_powf(
  // OGCG: call float @llvm.pow.f32(
}

double my_pow(double x, double y) {
  return __builtin_pow(x, y);
  // CIR: cir.func no_inline dso_local @my_pow
  // CIR: cir.pow %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_pow
  // LLVM:   call double @llvm.pow.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_pow(
  // OGCG: call double @llvm.pow.f64(
}

long double my_powl(long double x, long double y) {
  return __builtin_powl(x, y);
  // CIR: cir.func no_inline dso_local @my_powl
  // CIR: cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @my_powl
  // LLVM:   call x86_fp80 @llvm.pow.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_powl(
  // OGCG: call x86_fp80 @llvm.pow.f80(
}

float powf(float, float);
double pow(double, double);
long double powl(long double, long double);

float call_powf(float x, float y) {
  return powf(x, y);
  // CIR: cir.func no_inline dso_local @call_powf
  // CIR: cir.pow %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @call_powf
  // LLVM:   call float @llvm.pow.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_powf(
  // OGCG: call float @llvm.pow.f32(
}

double call_pow(double x, double y) {
  return pow(x, y);
  // CIR: cir.func no_inline dso_local @call_pow
  // CIR: cir.pow %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @call_pow
  // LLVM:   call double @llvm.pow.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_pow(
  // OGCG: call double @llvm.pow.f64(
}

long double call_powl(long double x, long double y) {
  return powl(x, y);
  // CIR: cir.func no_inline dso_local @call_powl
  // CIR: cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>

  // LLVM: define dso_local x86_fp80 @call_powl
  // LLVM:   call x86_fp80 @llvm.pow.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@call_powl(
  // OGCG: call x86_fp80 @llvm.pow.f80(
}

// acos

float my_acosf(float x) {
  return __builtin_acosf(x);
  // CIR: cir.func no_inline dso_local @my_acosf
  // CIR: cir.acos %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_acosf
  // LLVM:   call float @llvm.acos.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_acosf(
  // OGCG: call float @llvm.acos.f32(
}

double my_acos(double x) {
  return __builtin_acos(x);
  // CIR: cir.func no_inline dso_local @my_acos
  // CIR: cir.acos %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_acos
  // LLVM:   call double @llvm.acos.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_acos(
  // OGCG: call double @llvm.acos.f64(
}

// asin

float my_asinf(float x) {
  return __builtin_asinf(x);
  // CIR: cir.func no_inline dso_local @my_asinf
  // CIR: cir.asin %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_asinf
  // LLVM:   call float @llvm.asin.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_asinf(
  // OGCG: call float @llvm.asin.f32(
}

double my_asin(double x) {
  return __builtin_asin(x);
  // CIR: cir.func no_inline dso_local @my_asin
  // CIR: cir.asin %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_asin
  // LLVM:   call double @llvm.asin.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_asin(
  // OGCG: call double @llvm.asin.f64(
}

// atan

float my_atanf(float x) {
  return __builtin_atanf(x);
  // CIR: cir.func no_inline dso_local @my_atanf
  // CIR: cir.atan %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_atanf
  // LLVM:   call float @llvm.atan.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_atanf(
  // OGCG: call float @llvm.atan.f32(
}

double my_atan(double x) {
  return __builtin_atan(x);
  // CIR: cir.func no_inline dso_local @my_atan
  // CIR: cir.atan %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_atan
  // LLVM:   call double @llvm.atan.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_atan(
  // OGCG: call double @llvm.atan.f64(
}

// atan2

float my_atan2f(float y, float x) {
  return __builtin_atan2f(y, x);
  // CIR: cir.func no_inline dso_local @my_atan2f
  // CIR: cir.atan2 %{{.+}}, %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_atan2f
  // LLVM:   call float @llvm.atan2.f32(float %{{.+}}, float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_atan2f(
  // OGCG: call float @llvm.atan2.f32(
}

double my_atan2(double y, double x) {
  return __builtin_atan2(y, x);
  // CIR: cir.func no_inline dso_local @my_atan2
  // CIR: cir.atan2 %{{.+}}, %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_atan2
  // LLVM:   call double @llvm.atan2.f64(double %{{.+}}, double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_atan2(
  // OGCG: call double @llvm.atan2.f64(
}

// roundeven

float my_roundevenf(float x) {
  return __builtin_roundevenf(x);
  // CIR: cir.func no_inline dso_local @my_roundevenf
  // CIR: cir.roundeven %{{.+}} : !cir.float

  // LLVM: define dso_local float @my_roundevenf
  // LLVM:   call float @llvm.roundeven.f32(float %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_roundevenf(
  // OGCG: call float @llvm.roundeven.f32(
}

double my_roundeven(double x) {
  return __builtin_roundeven(x);
  // CIR: cir.func no_inline dso_local @my_roundeven
  // CIR: cir.roundeven %{{.+}} : !cir.double

  // LLVM: define dso_local double @my_roundeven
  // LLVM:   call double @llvm.roundeven.f64(double %{{.+}})
  // LLVM: }

  // OGCG: define{{.*}}@my_roundeven(
  // OGCG: call double @llvm.roundeven.f64(
}
