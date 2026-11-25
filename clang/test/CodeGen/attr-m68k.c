// RUN: %clang_cc1 -triple m68k-unknown-linux -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

__attribute__((interrupt(4))) void isr(void) {}

// CHECK: @__isr_2 = alias void (), ptr @isr
// CHECK: define{{.*}} cc101 void @isr() #0
// CHECK: attributes #0 = { nounwind {{.*}}}
