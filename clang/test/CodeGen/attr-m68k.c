// RUN: %clang_cc1 -triple m68k-unknown-linux -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

__attribute__((interrupt(4))) void isr(void) {}

__attribute__((interrupt(6))) void extern_isr(void);
void call_extern_isr(void) { extern_isr(); }

// CHECK: @__isr_2 = alias void (), ptr @isr
// CHECK-NOT: alias
// CHECK: define{{.*}} cc101 void @isr() #0
// CHECK: declare cc101 void @extern_isr()
// CHECK: attributes #0 = { nounwind {{.*}}}
