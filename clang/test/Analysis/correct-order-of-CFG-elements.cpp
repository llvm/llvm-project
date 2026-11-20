// RUN: %clang_analyze_cc1 -fcxx-exceptions -fexceptions -analyzer-checker=debug.DumpCFG -analyzer-config cfg-lifetime=true,cfg-scopes=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// Check the correct order of the CFG elements when AddLifetime is enabled.
// The CFG elements in a basic block should appear in reverse order of
// declaration. This test file also enables cfg-scopes to ensure that
// none of the elements appear as a duplicate in the given basic block.

struct NonTrivial {
  ~NonTrivial();
};

struct Trivial {
  int x;
};

// CHECK-LABEL: void test_trivial_and_nontrivial_order()
// CHECK-NEXT:  [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT:  [B1]
// CHECK-NEXT:    1: CFGScopeBegin(a)
// CHECK-NEXT:    2:  (CXXConstructExpr, [B1.3], NonTrivial)
// CHECK-NEXT:    3: NonTrivial a;
// CHECK-NEXT:    4: 0
// CHECK-NEXT:    5: int i = 0;
// CHECK-NEXT:    6:  (CXXConstructExpr, [B1.7], NonTrivial)
// CHECK-NEXT:    7: NonTrivial b;
// CHECK-NEXT:    8:  (CXXConstructExpr, [B1.9], Trivial)
// CHECK-NEXT:    9: Trivial u;
// CHECK-NEXT:   10: [B1.9] (Lifetime ends)
// CHECK-NEXT:   11: [B1.7].~NonTrivial() (Implicit destructor)
// CHECK-NEXT:   12: [B1.7] (Lifetime ends)
// CHECK-NEXT:   13: [B1.5] (Lifetime ends)
// CHECK-NEXT:   14: [B1.3].~NonTrivial() (Implicit destructor)
// CHECK-NEXT:   15: [B1.3] (Lifetime ends)
// CHECK-NEXT:   16: CFGScopeEnd(a)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT:  [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
void test_trivial_and_nontrivial_order() {
  NonTrivial a;
  int i = 0;
  NonTrivial b;
  Trivial u;
}

// CHECK-LABEL: void test_nested_scope()
// CHECK-NEXT:  [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK-EMPTY:
// CHECK-NEXT:  [B1]
// CHECK-NEXT:    1: CFGScopeBegin(a)
// CHECK-NEXT:    2:  (CXXConstructExpr, [B1.3], A)
// CHECK-NEXT:    3: A a;
// CHECK-NEXT:    4: CFGScopeBegin(c)
// CHECK-NEXT:    5:  (CXXConstructExpr, [B1.6], A)
// CHECK-NEXT:    6: A c;
// CHECK-NEXT:    7:  (CXXConstructExpr, [B1.8], A)
// CHECK-NEXT:    8: A d;
// CHECK-NEXT:    9: [B1.8] (Lifetime ends)
// CHECK-NEXT:   10: [B1.6] (Lifetime ends)
// CHECK-NEXT:   11: CFGScopeEnd(c)
// CHECK-NEXT:   12:  (CXXConstructExpr, [B1.13], A)
// CHECK-NEXT:   13: A b;
// CHECK-NEXT:   14: [B1.13] (Lifetime ends)
// CHECK-NEXT:   15: [B1.3] (Lifetime ends)
// CHECK-NEXT:   16: CFGScopeEnd(a)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK-EMPTY:
// CHECK-NEXT:  [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
struct A {};
void test_nested_scope() {
  A a;
  {
    A c;
    A d;
  }
  A b;
}
