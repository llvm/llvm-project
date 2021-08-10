// REQUIRES: x86-registered-target
/// Check asm because we use llvm::TargetOptions.

// RUN: %clang_cc1 -triple=x86_64 -S %s -falign-loops=8 -O -o - | FileCheck %s --check-prefixes=CHECK,CHECK_8
// RUN: %clang_cc1 -triple=x86_64 -S %s -falign-loops=32 -O -o - | FileCheck %s --check-prefixes=CHECK,CHECK_32

// CHECK-LABEL: foo:
// CHECK_8:       .p2align 3, 0x90
// CHECK_32:      .p2align 5, 0x90

void bar();
void foo() {
  for (int i = 0; i < 64; ++i)
    bar();
}
