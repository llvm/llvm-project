// RUN: %clang_cc1 -triple aarch64-windows-gnu -Oz -emit-llvm %s -o - | FileCheck %s

void *test_sponentry() {
  return __builtin_sponentry();
}
// CHECK-LABEL: define dso_local i8* @test_sponentry()
// CHECK: = tail call i8* @llvm.sponentry.p0i8()
// CHECK: ret i8*
