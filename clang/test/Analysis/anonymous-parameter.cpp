// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++20 %s 2>&1 | FileCheck %s

struct A {
  static A a;
  char b;
  friend bool operator==(A, A) = default;
};
bool _ = A() == A::a;

// FIXME: steps 1 and 5 show anonymous function parameters are
// not handled correctly.

// CHECK-LABEL: bool operator==(A, A) noexcept = default
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK:      [B1]
// CHECK-NEXT:    1: function-parameter-0-0
// CHECK-NEXT:    2: [B1.1].b
// CHECK-NEXT:    3: [B1.2] (ImplicitCastExpr, LValueToRValue, char)
// CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:    5: function-parameter-0-1
// CHECK-NEXT:    6: [B1.5].b
// CHECK-NEXT:    7: [B1.6] (ImplicitCastExpr, LValueToRValue, char)
// CHECK-NEXT:    8: [B1.7] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:    9: [B1.4] == [B1.8]
// CHECK-NEXT:   10: return [B1.9];
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
