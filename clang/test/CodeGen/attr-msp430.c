// RUN: %clang_cc1 -triple msp430-unknown-unknown -emit-llvm < %s| FileCheck %s

__attribute__((interrupt(1))) void foo(void) {}
// CHECK: @llvm.used
// CHECK-SAME: @foo

// CHECK: define msp430_intrcc void @foo() #0
// CHECK: attributes #0
// CHECK-SAME: noinline
// CHECK-SAME: "interrupt"="1"
