// RUN: %clang_cc1 -triple msp430-unknown-unknown -emit-llvm < %s| FileCheck %s
// RUN: %clang_cc1 -triple msp430-unknown-unknown -O2 -disable-llvm-passes -emit-llvm < %s | FileCheck --check-prefix=OPT %s

__attribute__((interrupt(1))) void foo(void) {}
// CHECK: @llvm.compiler.used =
// CHECK-SAME: @foo

// CHECK: define{{.*}} msp430_intrcc void @foo() #0
// CHECK: attributes #0
// CHECK-SAME: noinline
// CHECK-SAME: "interrupt"="1"

// OPT: attributes #0 = { nounwind "interrupt"="1" {{.*}}}
