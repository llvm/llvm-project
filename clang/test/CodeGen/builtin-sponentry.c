// RUN: %clang_cc1 -triple aarch64-windows-gnu -Oz -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7-windows-gnu -Oz -emit-llvm %s -o - | FileCheck %s

void *test_sponentry(void) {
  return __builtin_sponentry();
}
// CHECK-LABEL: define dso_local {{(arm_aapcs_vfpcc )?}}ptr @test_sponentry()
// CHECK: = tail call ptr @llvm.sponentry.p0()
// CHECK: ret ptr
