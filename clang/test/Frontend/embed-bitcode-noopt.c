// Ensure calling bypassing the driver with -fembed-bitcode embeds bitcode pre-
// optimizations
// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -O2 -emit-obj -triple=x86_64-unknown-linux-gnu -o %t.o -fembed-bitcode=all
// RUN: llvm-objcopy --dump-section=.llvmbc=%t.bc %t.o /dev/null

// Also check that the .llvmcmd section captures the optimization options.
// RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=CHECK-BC
// RUN: llvm-objcopy --dump-section=.llvmcmd=- %t.o /dev/null | FileCheck %s --check-prefix=CHECK-CMD

// CHECK-BC-LABEL: define void @bar() #0 {
// CHECK-BC-NEXT: entry:
// CHECK-BC-NEXT: ret void
// CHECK-BC-NEXT: }
// CHECK-BC-LABEL: define void @foo() #1 {
// CHECK-BC-NEXT: entry:
// CHECK-BC-NEXT: call void @bar()
// CHECK-BC-NEXT: ret void
// CHECK-BC-NEXT: }
// CHECK-BC-LABEL: attributes #0 = {{.*}} alwaysinline
// CHECK-CMD: -O2

__attribute__((always_inline)) void bar() {
  return;
}

void foo() {
  bar();
  return;
}
