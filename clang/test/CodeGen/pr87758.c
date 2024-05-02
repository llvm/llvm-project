// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O3 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O3 -fmath-errno -ffp-contract=on -fno-rounding-math -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffast-math -ffp-contract=fast -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -O3 -S -emit-llvm -Xclang -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang -O3 -ffp-model=fast -S -emit-llvm -Xclang -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang -O3 -ffp-model=precise -S -emit-llvm -Xclang -disable-llvm-passes %s -o - | FileCheck %s

// Reproducer for issue #87758
// The testcase below verifies that the "fast" flag are set on the calls.

float sqrtf(float x); // unary fp builtin
float powf(float x, float y); // binary fp builtin
float fmaf(float x, float y, float z); // ternary fp builtin
char *rindex(const char *s, int c); // not a fp builtin

#pragma float_control(push)
#pragma float_control(precise, off)
// CHECK-LABEL: define
// CHECK-SAME:  float @fp_precise_off_libm_calls(
// CHECK: %{{.*}} = call fast float @llvm.sqrt.f32(
// CHECK: %{{.*}} = call fast float @llvm.pow.f32(
// CHECK: %{{.*}} = call fast float @llvm.fma.f32(
// CHECK: %{{.*}} = call ptr @rindex(

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
// CHECK-LABEL: define
// CHECK-SAME:  float @fp_precise_on_libm_calls(
// CHECK: %{{.*}} = call
// CHECK-NOT: fast
// CHECK-SAME: float @sqrtf(
// CHECK: %{{.*}} = call
// CHECK-NOT: fast
// CHECK-SAME: float @powf(
// CHECK: %{{.*}} = call float @llvm.fma.f32(
// CHECK: %{{.*}} = call ptr @rindex(

float fp_precise_on_libm_calls(float a, float b, float c, const char *d, char *e, unsigned char f) {
  a = sqrtf(a);
  a = powf(a,b);
  a = fmaf(a,b,c);
  e = rindex(d, 75);
  return a;
}
#pragma float_control(pop)
