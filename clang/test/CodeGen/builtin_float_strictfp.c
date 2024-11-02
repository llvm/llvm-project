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
  // NOFP16:       [[LDADDR:%.*]] = load ptr, ptr %{{.*}}, align 8
  // NOFP16-NEXT:  [[IHALF:%.*]]  = load i16, ptr [[LDADDR]], align 2
  // NOFP16-NEXT:  [[CONV:%.*]]   = call float @llvm.convert.from.fp16.f32(i16 [[IHALF]])
  // NOFP16-NEXT:  [[IFLOAT:%.*]] = bitcast float [[CONV]] to i32
  // NOFP16-NEXT:  [[SHL:%.*]]    = shl i32 [[IFLOAT]], 1
  // NOFP16-NEXT:  [[RES1:%.*]]   = icmp eq i32 [[SHL]], -16777216
  // NOFP16-NEXT:                   zext i1 [[RES1]] to i32
  // FP16:         [[LDADDR:%.*]] = load ptr, ptr %{{.*}}, align 8
  // FP16-NEXT:    [[HALF:%.*]]   = load half, ptr [[LDADDR]], align 2
  // FP16-NEXT:    [[IHALF:%.*]]  = bitcast half [[HALF]] to i16
  // FP16-NEXT:    [[SHL:%.*]]    = shl i16 [[IHALF]], 1
  // FP16-NEXT:    [[RES1:%.*]]   = icmp eq i16 [[SHL]], -2048
  // FP16-NEXT:                     zext i1 [[RES1]] to i32
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
