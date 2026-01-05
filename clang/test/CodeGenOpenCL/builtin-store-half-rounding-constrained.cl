// RUN: %clang_cc1 %s -cl-std=cl3.0 -triple x86_64-unknown-unknown -disable-llvm-passes -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: @test_store_float(
// CHECK:         [[TMP0:%.*]] = call half @llvm.experimental.constrained.fptrunc.f16.f32(float {{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore")
// CHECK-NEXT:    store half [[TMP0]], ptr {{.*}}, align 2
// CHECK-NEXT:    ret void
//
__kernel void test_store_float(float foo, __global half* bar) {
  #pragma STDC FENV_ROUND FE_UPWARD
  __builtin_store_halff(foo, bar);
}

// CHECK-LABEL: @test_store_double(
// CHECK:         [[TMP0:%.*]] = call half @llvm.experimental.constrained.fptrunc.f16.f64(double {{.*}}, metadata !"round.downward", metadata !"fpexcept.ignore")
// CHECK-NEXT:    store half [[TMP0]], ptr {{.*}}, align 2
// CHECK-NEXT:    ret void
//
__kernel void test_store_double(double foo, __global half* bar) {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  __builtin_store_half(foo, bar);
}
