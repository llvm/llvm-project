// RUN: %clang_cc1 -emit-llvm-only -fdump-record-layouts-complete %s | FileCheck %s

struct a {
  int x;
};

struct b {
  char y;
} foo;

class c {};

class d;

// CHECK:          0 | struct a
// CHECK:          0 | struct b
// CHECK:          0 | class c
// CHECK-NOT:      0 | class d
