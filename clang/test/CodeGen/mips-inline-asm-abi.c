// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-linux-gnu -emit-obj -o - %s | \
// RUN:   llvm-readobj -h - | FileCheck %s

// CHECK: EF_MIPS_ABI_O32

__asm__(
"bar:\n"
"  nop\n"
);

void foo() {}
