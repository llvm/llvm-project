// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-lifetime=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// Tests for lifetime-end CFG nodes.

void test_simple_variable() {
  int i = 0;
}
// CHECK:      void test_simple_variable()
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: [B1]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: int i = 0;
// CHECK-NEXT:    3: [B1.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: ~A() noexcept
// CHECK-NEXT: [B1 (ENTRY)]
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
// CHECK-EMPTY:

struct A {
  ~A() {}
};
void test_nontrivial_dtor() {
  A a;
}
// CHECK-NEXT: void test_nontrivial_dtor()
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: [B1]
// CHECK-NEXT:    1:  (CXXConstructExpr, [B1.2], A)
// CHECK-NEXT:    2: A a;
// CHECK-NEXT:    3: [B1.2].~A() (Implicit destructor)
// CHECK-NEXT:    4: [B1.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
// CHECK-EMPTY:

void test_multiple_variables_nested_scopes() {
  int a = 0;
  int b = 0;
  {
    int c = 0, d = 0;
  }
}
// CHECK-NEXT: void test_multiple_variables_nested_scopes()
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: [B1]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: int a = 0;
// CHECK-NEXT:    3: 0
// CHECK-NEXT:    4: int b = 0;
// CHECK-NEXT:    5: 0
// CHECK-NEXT:    6: int c = 0;
// CHECK-NEXT:    7: 0
// CHECK-NEXT:    8: int d = 0;
// CHECK-NEXT:    9: [B1.8] (Lifetime ends)
// CHECK-NEXT:   10: [B1.6] (Lifetime ends)
// CHECK-NEXT:   11: [B1.4] (Lifetime ends)
// CHECK-NEXT:   12: [B1.2] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
// CHECK-EMPTY:

void test_local_static() {
  static int i = 0;
  int j = 0;
}
// CHECK-NEXT: void test_local_static()
// CHECK-NEXT: [B4 (ENTRY)]
// CHECK-NEXT:    Succs (1): B3
// CHECK-EMPTY:
// CHECK-NEXT: [B1]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: int j = 0;
// CHECK-NEXT:    3: [B1.2] (Lifetime ends)
// CHECK-NEXT:    Preds (2): B2 B3
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B2]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: static int i = 0;
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: [B3]
// CHECK-NEXT:    T: static init i
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (2): B1 B2
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
// CHECK-EMPTY:

void test_loop_body() {
  while (true) {
    int i = 0;
    break;
  }
}
// CHECK-NEXT: void test_loop_body()
// CHECK-NEXT: [B5 (ENTRY)]
// CHECK-NEXT:    Succs (1): B4
// CHECK-EMPTY:
// CHECK-NEXT: [B1]
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B4
// CHECK-EMPTY:
// CHECK-NEXT: [B2]
// CHECK-NEXT:    1: [B3.2] (Lifetime ends)
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: [B3]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: int i = 0;
// CHECK-NEXT:    3: [B3.2] (Lifetime ends)
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B4]
// CHECK-NEXT:    1: true
// CHECK-NEXT:    T: while [B4.1]
// CHECK-NEXT:    Preds (2): B1 B5
// CHECK-NEXT:    Succs (2): B3 NULL
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B3
// CHECK-EMPTY:

void test_lifetime_extended_temporary() {
  const int &r = 42;
}
// CHECK-NEXT: void test_lifetime_extended_temporary()
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: [B1]
// CHECK-NEXT:    1: 42
// CHECK-NEXT:    2: [B1.1] (ImplicitCastExpr, NoOp, const int)
// CHECK-NEXT:    3: [B1.2]
// CHECK-NEXT:    4: const int &r = 42;
// CHECK-NEXT:    5: [B1.4] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
