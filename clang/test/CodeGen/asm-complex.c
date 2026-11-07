// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

void f(_Complex float a, _Complex float b) {
  // CHECK-LABEL: @f(
  // CHECK: call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"(i64 %{{.*}})
  asm volatile("" :: "r"(a + b));
}
