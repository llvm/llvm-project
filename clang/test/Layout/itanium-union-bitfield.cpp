// RUN: %clang_cc1 -emit-llvm-only -triple %itanium_abi_triple -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s

// On z/OS, a bit-field has single byte alignment.  Add aligned(4) on z/OS so the union has
// the same size & alignment as expected.
#ifdef __MVS__
#define ALIGN4 __attribute__((aligned(4)))
#else
#define ALIGN4
#endif

union A {
  int f1 : 3 ALIGN4;
  A();
};

A::A() {}

union B {
  char f1 : 35 ALIGN4;
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
