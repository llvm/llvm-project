// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64be-unknown-elf -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=RV64
// RUN: %clang_cc1 -triple riscv32be-unknown-elf -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=RV32

// RV64: target datalayout = "E-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
// RV32: target datalayout = "E-m:e-p:32:32-i64:64-n32-S128"

int foo(void) {
  return 0;
}
