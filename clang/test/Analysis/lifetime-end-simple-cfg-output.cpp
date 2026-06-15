// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-lifetime=true,cfg-scopes=true %s > %t 2>&1
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
// CHECK-NEXT:    1: CFGScopeBegin(i)
// CHECK-NEXT:    2: 0
// CHECK-NEXT:    3: int i = 0;
// CHECK-NEXT:    4: [B1.3] (Lifetime ends)
// CHECK-NEXT:    5: CFGScopeEnd(i)
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
// CHECK-NEXT:    1: CFGScopeBegin(a)
// CHECK-NEXT:    2:  (CXXConstructExpr, [B1.3], A)
// CHECK-NEXT:    3: A a;
// CHECK-NEXT:    4: [B1.3].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B1.3] (Lifetime ends)
// CHECK-NEXT:    6: CFGScopeEnd(a)
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
// CHECK-NEXT:    1: CFGScopeBegin(a)
// CHECK-NEXT:    2: 0
// CHECK-NEXT:    3: int a = 0;
// CHECK-NEXT:    4: 0
// CHECK-NEXT:    5: int b = 0;
// CHECK-NEXT:    6: CFGScopeBegin(c)
// CHECK-NEXT:    7: 0
// CHECK-NEXT:    8: int c = 0;
// CHECK-NEXT:    9: 0
// CHECK-NEXT:   10: int d = 0;
// CHECK-NEXT:   11: [B1.10] (Lifetime ends)
// CHECK-NEXT:   12: [B1.8] (Lifetime ends)
// CHECK-NEXT:   13: CFGScopeEnd(c)
// CHECK-NEXT:   14: [B1.5] (Lifetime ends)
// CHECK-NEXT:   15: [B1.3] (Lifetime ends)
// CHECK-NEXT:   16: CFGScopeEnd(a)
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
// CHECK-NEXT:    1: CFGScopeBegin(j)
// CHECK-NEXT:    2: 0
// CHECK-NEXT:    3: int j = 0;
// CHECK-NEXT:    4: [B1.3] (Lifetime ends)
// CHECK-NEXT:    5: CFGScopeEnd(j)
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
// CHECK-NEXT:    1: [B3.3] (Lifetime ends)
// CHECK-NEXT:    2: CFGScopeEnd(i)
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT: [B3]
// CHECK-NEXT:    1: CFGScopeBegin(i)
// CHECK-NEXT:    2: 0
// CHECK-NEXT:    3: int i = 0;
// CHECK-NEXT:    4: [B3.3] (Lifetime ends)
// CHECK-NEXT:    5: CFGScopeEnd(i)
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
// CHECK-NEXT:    1: CFGScopeBegin(r)
// CHECK-NEXT:    2: 42
// CHECK-NEXT:    3: [B1.2] (ImplicitCastExpr, NoOp, const int)
// CHECK-NEXT:    4: [B1.3]
// CHECK-NEXT:    5: const int &r = 42;
// CHECK-NEXT:    6: [B1.5] (Lifetime ends)
// CHECK-NEXT:    7: CFGScopeEnd(r)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT: [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
