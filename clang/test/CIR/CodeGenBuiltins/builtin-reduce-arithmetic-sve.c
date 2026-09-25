// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sve \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sve \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sve \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
//
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

int test_sve_reduce_add(svint32_t x) {
  // CIR-LABEL: @test_sve_reduce_add
  // CIR: cir.call_llvm_intrinsic "vector.reduce.add"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_add
  // LLVM: call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_add(x);
}

int test_sve_reduce_mul(svint32_t x) {
  // CIR-LABEL: @test_sve_reduce_mul
  // CIR: cir.call_llvm_intrinsic "vector.reduce.mul"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_mul
  // LLVM: call i32 @llvm.vector.reduce.mul.nxv4i32(<vscale x 4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_mul(x);
}

int test_sve_reduce_max(svint32_t x) {
  // CIR-LABEL: @test_sve_reduce_max
  // CIR: cir.call_llvm_intrinsic "vector.reduce.smax"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_max
  // LLVM: call i32 @llvm.vector.reduce.smax.nxv4i32(<vscale x 4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_max(x);
}

unsigned test_sve_reduce_max_unsigned(svuint32_t x) {
  // CIR-LABEL: @test_sve_reduce_max_unsigned
  // CIR: cir.call_llvm_intrinsic "vector.reduce.umax"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_max_unsigned
  // LLVM: call i32 @llvm.vector.reduce.umax.nxv4i32(<vscale x 4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_max(x);
}

float test_sve_reduce_max_float(svfloat32_t x) {
  // CIR-LABEL: @test_sve_reduce_max_float
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmax"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_max_float
  // LLVM: call float @llvm.vector.reduce.fmax.nxv4f32(<vscale x 4 x float>
  // LLVM: ret float
  return __builtin_reduce_max(x);
}

int test_sve_reduce_min(svint32_t x) {
  // CIR-LABEL: @test_sve_reduce_min
  // CIR: cir.call_llvm_intrinsic "vector.reduce.smin"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_min
  // LLVM: call i32 @llvm.vector.reduce.smin.nxv4i32(<vscale x 4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_min(x);
}

unsigned test_sve_reduce_min_unsigned(svuint32_t x) {
  // CIR-LABEL: @test_sve_reduce_min_unsigned
  // CIR: cir.call_llvm_intrinsic "vector.reduce.umin"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_min_unsigned
  // LLVM: call i32 @llvm.vector.reduce.umin.nxv4i32(<vscale x 4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_min(x);
}

float test_sve_reduce_min_float(svfloat32_t x) {
  // CIR-LABEL: @test_sve_reduce_min_float
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmin"
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_min_float
  // LLVM: call float @llvm.vector.reduce.fmin.nxv4f32(<vscale x 4 x float>
  // LLVM: ret float
  return __builtin_reduce_min(x);
}

float test_sve_reduce_in_order_fadd(svfloat32_t x, float start) {
  // CIR-LABEL: @test_sve_reduce_in_order_fadd
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" {{.*}} : (!cir.float, !cir.vector<[4] x !cir.float>) -> !cir.float
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_in_order_fadd
  // LLVM: call float @llvm.vector.reduce.fadd.nxv4f32(float %{{.*}}, <vscale x 4 x float>
  // LLVM: ret float
  return __builtin_reduce_in_order_fadd(x, start);
}

float test_sve_reduce_in_order_fadd_cast_start(svfloat32_t x, double start) {
  // CIR-LABEL: @test_sve_reduce_in_order_fadd_cast_start
  // CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" {{.*}} : (!cir.float, !cir.vector<[4] x !cir.float>) -> !cir.float
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_in_order_fadd_cast_start
  // LLVM: fptrunc double %{{.*}} to float
  // LLVM: call float @llvm.vector.reduce.fadd.nxv4f32(float %{{.*}}, <vscale x 4 x float>
  // LLVM: ret float
  return __builtin_reduce_in_order_fadd(x, start);
}

double test_sve_reduce_in_order_fadd_double(svfloat64_t x, double start) {
  // CIR-LABEL: @test_sve_reduce_in_order_fadd_double
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" {{.*}} : (!cir.double, !cir.vector<[2] x !cir.double>) -> !cir.double
  // CIR: cir.return
  // LLVM-LABEL: @test_sve_reduce_in_order_fadd_double
  // LLVM: call double @llvm.vector.reduce.fadd.nxv2f64(double %{{.*}}, <vscale x 2 x double>
  // LLVM: ret double
  return __builtin_reduce_in_order_fadd(x, start);
}
