// REQUIRES: x86-registered-target
// -mllvm options of per-context libraries reach the LLVMContext that CIR is
// lowered into.
// RUN: %clang_cc1 -triple x86_64 -fclangir -S -mllvm -x86-asm-syntax=intel %s -o - | FileCheck %s

// CHECK: .intel_syntax
void f(void) {}
