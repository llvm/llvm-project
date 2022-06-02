// RUN: %clang_cc1 -no-opaque-pointers -triple aarch64-windows-gnu -Oz -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv7-windows-gnu -Oz -emit-llvm %s -o - | FileCheck %s

void *test_sponentry(void) {
  return __builtin_sponentry();
}
// CHECK-LABEL: define dso_local {{(arm_aapcs_vfpcc )?}}i8* @test_sponentry()
// CHECK: = tail call i8* @llvm.sponentry.p0i8()
// CHECK: ret i8*
