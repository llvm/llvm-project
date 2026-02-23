// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s

// Test that large _BitInt fields do not overflow the internal storage unit tracker (previously unsigned char).
// If the bug exists, this struct splits into two units.
// If fixed, f1 and f2 are packed into a single unit.

#pragma ms_struct on

struct __attribute__((packed, aligned(1))) A {
  _BitInt(250) f1 : 2;
  _BitInt(250) f2 : 2;
};

struct __attribute__((packed, aligned(1))) B {
  _BitInt(500) f1 : 2;
  _BitInt(500) f2 : 255;
};

// CHECK-LABEL:   0 | struct A{{$}}
// CHECK-NEXT:0:0-1 |   _BitInt(250) f1
// CHECK-NEXT:0:2-3 |   _BitInt(250) f2
// CHECK-NEXT:      | [sizeof=32, align=32]

// CHECK-LABEL:   0 | struct B{{$}}
// CHECK-NEXT:0:0-1 |   _BitInt(500) f1
// CHECK-NEXT:0:2-256 |   _BitInt(500) f2
// CHECK-NEXT:      | [sizeof=64, align=64]

int x[sizeof(struct A)];
int y[sizeof(struct B)];
