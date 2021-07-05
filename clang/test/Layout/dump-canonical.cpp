// RUN: %clang_cc1 -emit-llvm-only -fdump-record-layouts %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm-only -fdump-record-layouts-canonical %s | FileCheck %s -check-prefix CANONICAL

typedef long foo_t;


struct a {
  foo_t x;
} b;

struct c {
  typedef foo_t bar_t;
  bar_t x;
} d;

// CHECK:          0 | foo_t
// CHECK:          0 | c::bar_t
// CANONICAL-NOT:  0 | foo_t
// CANONICAL-NOT:  0 | c::bar_t
// CANONICAL:      0 | long
