// REQUIRES: x86-registered-target
/// Some clang-cl users expect AT&T syntax input even if -x86-asm-syntax=intel is set.
// RUN: %clang_cc1 -triple x86_64-windows-msvc -S -fms-extensions -mllvm -x86-asm-syntax=intel %s -o - | FileCheck %s

// CHECK:         .intel_syntax noprefix
// CHECK:         mov     rax, rax
// CHECK-LABEL: foo:
// CHECK:         mov     rdx, rdx
// CHECK:         mov     rdx, rdx

asm("movq %rax, %rax");

void foo() {
  asm("movq %rdx, %rdx");

  __asm {
    mov rdx, rdx
  }
}
