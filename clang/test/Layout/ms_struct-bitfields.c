// RUN: %clang_cc1 -emit-llvm-only -triple armv7-apple-darwin -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s

// rdar://22275433

#pragma ms_struct on

union A {
  unsigned long long x : 32;
  unsigned long long y : 32;
} a;
// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:        0 | union A
// CHECK-NEXT:   0:0-31 |   unsigned long long x
// CHECK-NEXT:   0:0-31 |   unsigned long long y
// CHECK-NEXT:          | [sizeof=8, align=1]

union B {
  __attribute__((aligned(4)))
  unsigned long long x : 32;
  unsigned long long y : 32;
} b;
// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:       0 | union B
// CHECK-NEXT:  0:0-31 |   unsigned long long x
// CHECK-NEXT:  0:0-31 |   unsigned long long y
// CHECK-NEXT:         | [sizeof=8, align=1]

union C {
  unsigned long long : 0;
  unsigned short y : 8;
} c;
// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:       0 | union C
// CHECK-NEXT:     0:- |   unsigned long long
// CHECK-NEXT:   0:0-7 |   unsigned short y
// CHECK-NEXT:         | [sizeof=2, align=1]

union D {
  unsigned long long : 0;
  unsigned short : 0;
} d;
// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:       0 | union D
// CHECK-NEXT:     0:- |   unsigned long long
// CHECK-NEXT:     0:- |   unsigned short
// CHECK-NEXT:         | [sizeof=1, align=1]

