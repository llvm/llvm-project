// RUN: %clang_cc1 -emit-llvm-only -triple %itanium_abi_triple -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s

union A {
  int f1: 3;
  A();
};

A::A() {}

union B {
  char f1: 35;
  B();
};

B::B() {}

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:     0 | union A
// CHECK-NEXT: 0:0-2 |   int f1
// CHECK-NEXT:       | [sizeof=4, dsize=1, align=4{{(, preferredalign=4,)?}}
// CHECK-NEXT:       |  nvsize=1, nvalign=4{{(, preferrednvalign=4)?}}]

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | union B
// CHECK-NEXT: 0:0-34 |   char f1
// CHECK-NEXT:        | [sizeof=8, dsize=5, align=4{{(, preferredalign=4,)?}}
// CHECK-NEXT:        |  nvsize=5, nvalign=4{{(, preferrednvalign=4)?}}]
