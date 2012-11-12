// RUN: %clang_cc1 -fmath-errno -emit-llvm -o - %s -triple i386-unknown-unknown | FileCheck -check-prefix YES %s
// RUN: %clang_cc1 -emit-llvm -o - %s -triple i386-unknown-unknown | FileCheck -check-prefix NO %s

// CHECK-YES: define void @test_sqrt
// CHECK-NO: define void @test_sqrt
void test_sqrt(float a0, double a1, long double a2) {
  // Following llvm-gcc's lead, we never emit these as intrinsics;
  // no-math-errno isn't good enough.  We could probably use intrinsics
  // with appropriate guards if it proves worthwhile.

  // CHECK-YES: call float @sqrtf
  // CHECK-NO: call float @sqrtf
  float l0 = sqrtf(a0);

  // CHECK-YES: call double @sqrt
  // CHECK-NO: call double @sqrt
  double l1 = sqrt(a1);

  // CHECK-YES: call x86_fp80 @sqrtl
  // CHECK-NO: call x86_fp80 @sqrtl
  long double l2 = sqrtl(a2);
}

// CHECK-YES: declare float @sqrtf(float)
// CHECK-YES: declare double @sqrt(double)
// CHECK-YES: declare x86_fp80 @sqrtl(x86_fp80)
// CHECK-NO: declare float @sqrtf(float) nounwind readnone
// CHECK-NO: declare double @sqrt(double) nounwind readnone
// CHECK-NO: declare x86_fp80 @sqrtl(x86_fp80) nounwind readnone

// CHECK-YES: define void @test_pow
// CHECK-NO: define void @test_pow
void test_pow(float a0, double a1, long double a2) {
  // CHECK-YES: call float @powf
  // CHECK-NO: call float @llvm.pow.f32
  float l0 = powf(a0, a0);

  // CHECK-YES: call double @pow
  // CHECK-NO: call double @llvm.pow.f64
  double l1 = pow(a1, a1);

  // CHECK-YES: call x86_fp80 @powl
  // CHECK-NO: call x86_fp80 @llvm.pow.f80
  long double l2 = powl(a2, a2);
}

// CHECK-YES: declare float @powf(float, float)
// CHECK-YES: declare double @pow(double, double)
// CHECK-YES: declare x86_fp80 @powl(x86_fp80, x86_fp80)
// CHECK-NO: declare float @llvm.pow.f32(float, float) nounwind readonly
// CHECK-NO: declare double @llvm.pow.f64(double, double) nounwind readonly
// CHECK-NO: declare x86_fp80 @llvm.pow.f80(x86_fp80, x86_fp80) nounwind readonly

// CHECK-YES: define void @test_fma
// CHECK-NO: define void @test_fma
void test_fma(float a0, double a1, long double a2) {
    // CHECK-YES: call float @llvm.fma.f32
    // CHECK-NO: call float @llvm.fma.f32
    float l0 = fmaf(a0, a0, a0);

    // CHECK-YES: call double @llvm.fma.f64
    // CHECK-NO: call double @llvm.fma.f64
    double l1 = fma(a1, a1, a1);

    // CHECK-YES: call x86_fp80 @llvm.fma.f80
    // CHECK-NO: call x86_fp80 @llvm.fma.f80
    long double l2 = fmal(a2, a2, a2);
}

// CHECK-YES: declare float @llvm.fma.f32(float, float, float) nounwind readnone
// CHECK-YES: declare double @llvm.fma.f64(double, double, double) nounwind readnone
// CHECK-YES: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) nounwind readnone
// CHECK-NO: declare float @llvm.fma.f32(float, float, float) nounwind readnone
// CHECK-NO: declare double @llvm.fma.f64(double, double, double) nounwind readnone
// CHECK-NO: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) nounwind readnone

// Just checking to make sure these library functions are marked readnone
void test_builtins(double d, float f, long double ld) {
// CHEC-NO: @test_builtins
// CHEC-YES: @test_builtins
  double atan_ = atan(d);
  long double atanl_ = atanl(ld);
  float atanf_ = atanf(f);
// CHECK-NO: declare double @atan(double) nounwind readnone
// CHECK-NO: declare x86_fp80 @atanl(x86_fp80) nounwind readnone
// CHECK-NO: declare float @atanf(float) nounwind readnone
// CHECK-YES-NOT: declare double @atan(double) nounwind readnone
// CHECK-YES-NOT: declare x86_fp80 @atanl(x86_fp80) nounwind readnone
// CHECK-YES-NOT: declare float @atanf(float) nounwind readnone

  double atan2_ = atan2(d, 2);
  long double atan2l_ = atan2l(ld, ld);
  float atan2f_ = atan2f(f, f);
// CHECK-NO: declare double @atan2(double, double) nounwind readnone
// CHECK-NO: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) nounwind readnone
// CHECK-NO: declare float @atan2f(float, float) nounwind readnone
// CHECK-YES-NOT: declare double @atan2(double, double) nounwind readnone
// CHECK-YES-NOT: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) nounwind readnone
// CHECK-YES-NOT: declare float @atan2f(float, float) nounwind readnone

  double exp_ = exp(d);
  long double expl_ = expl(ld);
  float expf_ = expf(f);
// CHECK-NO: declare double @exp(double) nounwind readnone
// CHECK-NO: declare x86_fp80 @expl(x86_fp80) nounwind readnone
// CHECK-NO: declare float @expf(float) nounwind readnone
// CHECK-YES-NOT: declare double @exp(double) nounwind readnone
// CHECK-YES-NOT: declare x86_fp80 @expl(x86_fp80) nounwind readnone
// CHECK-YES-NOT: declare float @expf(float) nounwind readnone

  double log_ = log(d);
  long double logl_ = logl(ld);
  float logf_ = logf(f);
// CHECK-NO: declare double @log(double) nounwind readnone
// CHECK-NO: declare x86_fp80 @logl(x86_fp80) nounwind readnone
// CHECK-NO: declare float @logf(float) nounwind readnone
// CHECK-YES-NOT: declare double @log(double) nounwind readnone
// CHECK-YES-NOT: declare x86_fp80 @logl(x86_fp80) nounwind readnone
// CHECK-YES-NOT: declare float @logf(float) nounwind readnone
}
