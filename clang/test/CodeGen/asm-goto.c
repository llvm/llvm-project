// RUN: %clang_cc1 -O0 -emit-llvm  %s -o - | FileCheck %s

int foo(int cond)
{
  // CHECK: callbr void asm sideeffect
  // CHECK: to label %asm.fallthrough [label %label_true, label %loop], !srcloc !2
  // CHECK: asm.fallthrough:
  asm volatile goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
  asm volatile goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
  // CHECK: callbr void asm sideeffect
  // CHECK: to label %asm.fallthrough1 [label %label_true, label %loop], !srcloc !3
  // CHECK: asm.fallthrough1:
  return 0;
loop:
  return 0;
label_true:
  return 1;
}
