// RUN: %clang_cc1 -emit-llvm -triple x86_64-windows-pc -ffp-exception-behavior=maytrap -o - %s | FileCheck %s --check-prefixes=CHECK,FP16
// RUN: %clang_cc1 -emit-llvm -triple ppc64-be -ffp-exception-behavior=maytrap -o - %s | FileCheck %s --check-prefixes=CHECK,NOFP16

// test to ensure that these builtins don't do the variadic promotion of float->double.

// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)

// CHECK-LABEL: @test_half
void test_half(__fp16 *H, __fp16 *H2) {
  (void)__builtin_isgreater(*H, *H2);
  // FP16: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // FP16: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK-NEXT: zext i1
  (void)__builtin_isinf(*H);
  // NOFP16: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0x7FF0000000000000, metadata !"oeq", metadata !"fpexcept.strict")
  // FP16: call i1 @llvm.experimental.constrained.fcmp.f16(half %{{.*}}, half 0xH7C00, metadata !"oeq", metadata !"fpexcept.strict")
}

// CHECK-LABEL: @test_mixed
void test_mixed(double d1, float f2) {
  (void)__builtin_isgreater(d1, f2);
  // CHECK: [[CONV:%.*]] = call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK-NEXT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double [[CONV]], metadata !"ogt", metadata !"fpexcept.strict")
  // CHECK-NEXT: zext i1
  (void)__builtin_isgreaterequal(d1, f2);
  // CHECK: [[CONV:%.*]] = call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK-NEXT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double [[CONV]], metadata !"oge", metadata !"fpexcept.strict")
  // CHECK-NEXT: zext i1
  (void)__builtin_isless(d1, f2);
  // CHECK: [[CONV:%.*]] = call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK-NEXT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double [[CONV]], metadata !"olt", metadata !"fpexcept.strict")
  // CHECK-NEXT: zext i1
  (void)__builtin_islessequal(d1, f2);
  // CHECK: [[CONV:%.*]] = call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK-NEXT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double [[CONV]], metadata !"ole", metadata !"fpexcept.strict")
  // CHECK-NEXT: zext i1
  (void)__builtin_islessgreater(d1, f2);
  // CHECK: [[CONV:%.*]] = call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK-NEXT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double [[CONV]], metadata !"one", metadata !"fpexcept.strict")
  // CHECK-NEXT: zext i1
  (void)__builtin_isunordered(d1, f2);
  // CHECK: [[CONV:%.*]] = call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
  // CHECK-NEXT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double [[CONV]], metadata !"uno", metadata !"fpexcept.strict")
  // CHECK-NEXT: zext i1
}
