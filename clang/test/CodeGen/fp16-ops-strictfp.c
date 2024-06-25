// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - -triple arm-none-linux-gnueabi %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK -vv -dump-input=fail
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -emit-llvm -o - -triple aarch64 %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - -triple x86_64-linux-gnu %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - -triple arm-none-linux-gnueabi -fnative-half-type %s \
// RUN:   | FileCheck %s --check-prefix=NATIVE-HALF --check-prefix=CHECK
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -emit-llvm -o - -triple aarch64 -fnative-half-type %s \
// RUN:   | FileCheck %s --check-prefix=NATIVE-HALF --check-prefix=CHECK
//
// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)

typedef unsigned cond_t;
typedef __fp16 float16_t;

volatile cond_t test;
volatile int i0;
volatile __fp16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;
volatile double d0;
short s0;

void foo(void) {
  // CHECK-LABEL: define{{.*}} void @foo()

  // Check unary ops

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // NATIVE-HALF: call i32 @llvm.experimental.constrained.fptoui.i32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.uitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 = (test);

  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half 0xH0000, metadata !"une", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (!h1);

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: fneg float
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: fneg half
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = -h1;

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: load volatile half
  // NATIVE-HALF-NEXT: store volatile half
  // NOTNATIVE: store {{.*}} half {{.*}}, ptr
  h1 = +h1;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half 0xH3C00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1++;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half 0xH3C00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  ++h1;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half 0xHBC00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  --h1;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half 0xHBC00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1--;

  // Check binary ops with various operands
  // NATIVE-HALF: call half @llvm.experimental.constrained.fmul.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = h0 * h2;

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float -2.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fmul.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = h0 * (__fp16) -2.0f;

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = h0 * f2;

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = f0 * h2;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fmul.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = h0 * i0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fdiv.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h0 / h2);

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float -2.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fdiv.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h0 / (__fp16) -2.0f);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h0 / f2);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (f0 / h2);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fdiv.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h0 / i0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h2 + h0);

  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f64(double -2.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = ((__fp16)-2.0 + h0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h2 + f0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (f2 + h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h0 + i0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.fsub.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h2 - h0);

  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float -2.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fsub.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = ((__fp16)-2.0f - h0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h2 - f0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (f2 - h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fsub.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h0 - i0);

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h2 < h0);

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 4.200000e+01, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 4.200000e+01, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h2 < (__fp16)42.0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h2 < f0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (f2 < h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (i0 < h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 < i0);

  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 > h2);

  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 4.200000e+01, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 4.200000e+01, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = ((__fp16)42.0 > h2);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 > f2);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (f0 > h2);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (i0 > h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 > i0);

  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h2 <= h0);

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 4.200000e+01, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 4.200000e+01, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %98, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h2 <= (__fp16)42.0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h2 <= f0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (f2 <= h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (i0 <= h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 <= i0);


  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 >= h2);

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f64(double -2.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 >= (__fp16)-2.0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 >= f2);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (f0 >= h2);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (i0 >= h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 >= i0);

  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h1 == h2);

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %122, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h1 == (__fp16)1.0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h1 == f1);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (f1 == h1);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (i0 == h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 == i0);

  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h1 != h2);

  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h1 != (__fp16)1.0);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h1 != f1);

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (f1 != h1);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (i0 != h0);

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  test = (h0 != i0);

  // NATIVE-HALF: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half 0xH0000, metadata !"une", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float {{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h1 = (h1 ? h2 : h0);

  // Check assignments (inc. compound)
  // CHECK: store {{.*}} half {{.*}}, ptr
  // xATIVE-HALF: store {{.*}} half 0xHC000 // FIXME: We should be folding here.
  h0 = h1;

  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float -2.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 = (__fp16)-2.0f;

  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 = f0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 = i0;

  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // NATIVE-HALF: call i32 @llvm.experimental.constrained.fptosi.i32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  i0 = h0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 += h1;

  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f32(float 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 += (__fp16)1.0f;

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 += f2;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i32 @llvm.experimental.constrained.fptosi.i32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  i0 += h0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fadd.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 += i0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fsub.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 -= h1;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fsub.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 -= (__fp16)1.0;

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 -= f2;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fsub.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i32 @llvm.experimental.constrained.fptosi.i32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  i0 -= h0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fsub.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fsub.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 -= i0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fmul.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 *= h1;

  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fmul.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 *= (__fp16)1.0;

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 *= f2;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fmul.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i32 @llvm.experimental.constrained.fptosi.i32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  i0 *= h0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fmul.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 *= i0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.fdiv.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 /= h1;

  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fptrunc.f16.f64(double 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fdiv.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 /= (__fp16)1.0;

  // CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 /= f2;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fdiv.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call i32 @llvm.experimental.constrained.fptosi.i32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: store {{.*}} i32 {{.*}}, ptr
  i0 /= h0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NATIVE-HALF: call half @llvm.experimental.constrained.fdiv.f16(half %{{.*}}, half %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.fdiv.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 /= i0;

  // Check conversions to/from double
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 = d0;

  // CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 = (float)d0;

  // CHECK: call double @llvm.experimental.constrained.fpext.f64.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: store {{.*}} double {{.*}}, ptr
  d0 = h0;

  // CHECK: [[MID:%.*]] = call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call double @llvm.experimental.constrained.fpext.f64.f32(float [[MID]], metadata !"fpexcept.strict")
  // CHECK: store {{.*}} double {{.*}}, ptr
  d0 = (float)h0;

  // NATIVE-HALF: call half @llvm.experimental.constrained.sitofp.f16.i16(i16 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call float @llvm.experimental.constrained.sitofp.f32.i16(i16 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // NOTNATIVE: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: store {{.*}} half {{.*}}, ptr
  h0 = s0;
}

// CHECK-LABEL: define{{.*}} void @testTypeDef(
// NATIVE-HALF: call <4 x half> @llvm.experimental.constrained.fadd.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// NOTNATIVE: %[[CONV:.*]] = call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
// NOTNATIVE: %[[CONV1:.*]] = call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
// NOTNATIVE: %[[ADD:.*]] = call <4 x float> @llvm.experimental.constrained.fadd.v4f32(<4 x float> %conv, <4 x float> %conv1, metadata !"round.tonearest", metadata !"fpexcept.strict")
// NOTNATIVE: call <4 x half> @llvm.experimental.constrained.fptrunc.v4f16.v4f32(<4 x float> %add, metadata !"round.tonearest", metadata !"fpexcept.strict")

void testTypeDef() {
  __fp16 t0 __attribute__((vector_size(8)));
  float16_t t1 __attribute__((vector_size(8)));
  t1 = t0 + t1;
}

