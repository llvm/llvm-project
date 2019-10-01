// RUN: %clang_analyze_cc1  -triple i386-pc-linux-gnu -analyzer-checker=debug.DumpCFG %s 2>&1 | FileCheck %s
// RUN: %clang_analyze_cc1  -triple x86_64-pc-linux-gnu -analyzer-checker=debug.DumpCFG %s 2>&1 | FileCheck %s

int foo(int cond)
{
label_true:
  asm goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
  return 0;
loop:
  return 0;
}

// CHECK-LABEL: loop
// CHECK-NEXT: 0
// CHECK-NEXT: return
// CHECK-NEXT: Preds (1): B3
// CHECK-NEXT: Succs (1): B0

// CHECK-LABEL: label_true
// CHECK-NEXT: asm goto
// CHECK-NEXT: Preds (2): B3 B4
// CHECK-NEXT: Succs (3): B2 B3 B1


int bar(int cond)
{
  asm goto("testl %0, %0; jne %l1;" :: "r"(cond)::L1, L2);
  return 0;
L1:
L2:
  return 0;
}

// CHECK: [B4]
// CHECK-NEXT: asm goto
// CHECK-NEXT: Preds (1): B5
// CHECK-NEXT: Succs (3): B3 B2 B1

int zoo(int n)
{
A5:
A1:
  asm goto("testl %0, %0; jne %l1;" :: "r"(n)::A1, A2, A3, A4, A5);
A2:
A3:
A4:
  return 0;
}

// CHECK-LABEL: A1
// CHECK-NEXT: asm goto
// CHECK-NEXT: Preds (2): B5 B4
// CHECK-NEXT: Succs (5): B3 B4 B2 B1 B5
