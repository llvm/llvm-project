// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc \
// RUN:     -fexceptions -fcxx-exceptions | FileCheck %s

// Make sure calls to inline asm have funclet bundles.

extern "C" void might_throw();
extern "C" void foo() {
  try {
    might_throw();
  } catch (int) {
    __asm__("nop");
  }
}

// CHECK-LABEL: define void @foo()
// CHECK: invoke void @might_throw()
// CHECK: %[[CATCHPAD:[^ ]*]] = catchpad within
// CHECK: call void asm sideeffect "nop", {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
