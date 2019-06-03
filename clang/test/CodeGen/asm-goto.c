// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O0 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu-O0 -emit-llvm %s -o - | FileCheck %s

int foo(int cond)
{
  // CHECK: callbr void asm sideeffect
  // CHECK: to label %asm.fallthrough [label %label_true, label %loop], !srcloc
  // CHECK: asm.fallthrough:
  asm volatile goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
  asm volatile goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
  // CHECK: callbr void asm sideeffect
  // CHECK: to label %asm.fallthrough1 [label %label_true, label %loop], !srcloc
  // CHECK: asm.fallthrough1:
  return 0;
loop:
  return 0;
label_true:
  return 1;
}
