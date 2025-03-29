// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// precise mode
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fmath-errno -ffp-contract=on \
// RUN: -fno-rounding-math -emit-llvm  -o - %s | FileCheck \
// RUN: --check-prefix=CHECK-PRECISE %s

// fast mode
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffast-math -ffp-contract=fast \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-FAST %s

// Reproducer for issue #87758
// The testcase below verifies that the "fast" flag are set on the calls.

float sqrtf(float x); // unary fp builtin
float powf(float x, float y); // binary fp builtin
float fmaf(float x, float y, float z); // ternary fp builtin
char *rindex(const char *s, int c); // not a fp builtin

#pragma float_control(push)
#pragma float_control(precise, off)
// CHECK: define dso_local float @fp_precise_off_libm_calls(
// CHECK: call fast float @llvm.sqrt.f32(
// CHECK: call fast float @llvm.pow.f32(
// CHECK: call fast float @llvm.fma.f32(
// CHECK: call ptr @rindex(

// CHECK-PRECISE: define dso_local float @fp_precise_off_libm_calls(
// CHECK-PRECISE: call fast float @sqrtf(
// CHECK-PRECISE: call fast float @powf(
// CHECK-PRECISE: call fast float @llvm.fma.f32(
// CHECK-PRECISE: call ptr @rindex(

// CHECK-FAST: define dso_local nofpclass(nan inf) float @fp_precise_off_libm_calls(
// CHECK-FAST: call fast float @llvm.sqrt.f32(
// CHECK-FAST: call fast float @llvm.pow.f32(
// CHECK-FAST: call fast float @llvm.fma.f32(
// CHECK-FAST: call ptr @rindex(

float fp_precise_off_libm_calls(float a, float b, float c, const char *d, char *e, unsigned char f) {
  a = sqrtf(a);
  a = powf(a,b);
  a = fmaf(a,b,c);
  e = rindex(d, 75);
  return a;
}
#pragma float_control(pop)

#pragma float_control(push)
#pragma float_control(precise, on)
// CHECK: define dso_local float @fp_precise_on_libm_calls(
// CHECK: call float @sqrtf(
// CHECK: call float @powf(
// CHECK: call float @llvm.fma.f32(
// CHECK: call ptr @rindex(

// CHECK-PRECISE: define dso_local float @fp_precise_on_libm_calls(
// CHECK-PRECISE: call float @sqrtf(
// CHECK-PRECISE: call float @powf(
// CHECK-PRECISE: call float @llvm.fma.f32(
// CHECK-PRECISE: call ptr @rindex(

// CHECK-FAST: define dso_local nofpclass(nan inf) float @fp_precise_on_libm_calls(
// CHECK-FAST: call nofpclass(nan inf) float @sqrtf(
// CHECK-FAST: call nofpclass(nan inf) float @powf(
// CHECK-FAST: call float @llvm.fma.f32(
// CHECK-FAST: call ptr @rindex(

float fp_precise_on_libm_calls(float a, float b, float c, const char *d, char *e, unsigned char f) {
  a = sqrtf(a);
  a = powf(a,b);
  a = fmaf(a,b,c);
  e = rindex(d, 75);
  return a;
}
#pragma float_control(pop)
