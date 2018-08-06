// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only -fdump-record-layouts %s \
// RUN:            | FileCheck %s

struct S {
  char x;
  int y;
} __attribute__((packed, aligned(8)));

struct alignas(8) T {
  char x;
  int y;
} __attribute__((packed));

S s;
T t;
// CHECK:          0 | struct T
// CHECK-NEXT:          0 |   char x
// CHECK-NEXT:          1 |   int y
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=8]

// CHECK:          0 | struct S
// CHECK-NEXT:          0 |   char x
// CHECK-NEXT:          1 |   int y
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=8]
