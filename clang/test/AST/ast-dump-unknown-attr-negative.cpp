// Retention is limited to C++ [[...]] spellings. GNU __attribute__ and MS
// __declspec unknown attributes are dropped after their usual diagnostic, as
// before, not retained as UnknownAttr. This pins the scope of the feature.

// RUN: %clang_cc1 -std=c++20 -fms-extensions -Wno-unknown-attributes \
// RUN:   -Wno-ignored-attributes -ast-dump %s | FileCheck %s

__attribute__((unknown_gnu)) int a;
__declspec(unknown_ms) int b;
// Neither GNU nor __declspec unknowns are retained.
// CHECK-NOT: UnknownAttr

// A C++ [[...]] spelling is retained (positive control).
[[unknown_cxx]] int c;
// CHECK: VarDecl {{.*}} c 'int'
// CHECK-NEXT: UnknownAttr {{.*}} unknown_cxx
