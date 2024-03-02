// RUN: %clang_cc1 -fsyntax-only -fdump-record-layouts-complete %s | FileCheck %s

struct a {
  int x;
};

struct b {
  char y;
} foo;

class c {};

class d;

template <typename>
struct s {
  int x;
};

template <typename T>
struct ts {
  T x;
};

void f() {
  ts<int> a;
  ts<double> b;
}

namespace gh83684 {
template <class Pointer>
struct AllocationResult {
  Pointer ptr = nullptr;
  int count = 0;
};
}

// CHECK:          0 | struct a
// CHECK:          0 | struct b
// CHECK:          0 | class c
// CHECK:          0 | struct ts<int>
// CHECK:          0 | struct ts<double>
// CHECK-NOT:      0 | class d
// CHECK-NOT:      0 | struct s
// CHECK-NOT:      0 | struct AllocationResult
