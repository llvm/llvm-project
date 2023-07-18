// RUN: %clang_analyze_cc1 -std=c++11 -fcxx-exceptions -fexceptions -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=true,cfg-implicit-dtors=true,cfg-lifetime=true,cfg-scopes=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK %s

class A {
public:
  int x;
  [[noreturn]] ~A();
};

void foo();
extern const bool UV;

// CHECK:       [B3 (ENTRY)]
// CHECK-NEXT:    Succs (1): B2
//
// CHECK:       [B1]
// CHECK-NEXT:    1: CFGScopeEnd(a)
// CHECK-NEXT:    2: foo
// CHECK-NEXT:    3: [B1.2] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(void))
// CHECK-NEXT:    4: [B1.3]()
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B2 (NORETURN)]
// CHECK-NEXT:    1: CFGScopeBegin(a)
// CHECK-NEXT:    2:  (CXXConstructExpr, [B2.3], A)
// CHECK-NEXT:    3: A a;
// CHECK-NEXT:    4: [B2.3].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B2.3] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (2): B1 B2
void test_single_decl() {
  {
    A a;
  }
  foo();
}

// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:    Succs (1): B5
//
// CHECK:       [B1]
// CHECK-NEXT:   label:
// CHECK-NEXT:    1: foo
// CHECK-NEXT:    2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(void))
// CHECK-NEXT:    3: [B1.2]()
// CHECK-NEXT:    Preds (4): B2 B3(Unreachable) B4 B5(Unreachable)
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B2]
// CHECK-NEXT:    1: CFGScopeEnd(a)
// CHECK-NEXT:    Succs (1): B1
//
// CHECK:       [B3 (NORETURN)]
// CHECK-NEXT:    1: [B5.3].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B5.3] (Lifetime ends)
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B4]
// CHECK-NEXT:    1: CFGScopeEnd(a)
// CHECK-NEXT:    T: goto label;
// CHECK-NEXT:    Succs (1): B1
//
// CHECK:       [B5 (NORETURN)]
// CHECK-NEXT:    1: CFGScopeBegin(a)
// CHECK-NEXT:    2:  (CXXConstructExpr, [B5.3], A)
// CHECK-NEXT:    3: A a;
// CHECK-NEXT:    4: [B5.3].~A() (Implicit destructor)
// CHECK-NEXT:    5: [B5.3] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B6
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (3): B1 B3 B5
void test_forward_goto() {
  {
    A a;
    goto label;
  }
label:
  foo();
}


// The blocks B3 and B5, are inserted during backpatching goto stmt, to handle
// scope changes.
// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:    Succs (1): B3
//
// CHECK:       [B1]
// CHECK-NEXT:    1: CFGScopeEnd(a)
// CHECK-NEXT:    2: foo
// CHECK-NEXT:    3: [B1.2] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(void))
// CHECK-NEXT:    4: [B1.3]()
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B2 (NORETURN)]
// CHECK-NEXT:    1: [B3.3].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B3.3] (Lifetime ends)
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B3]
// CHECK-NEXT:   label:
// CHECK-NEXT:    1: CFGScopeBegin(a)
// CHECK-NEXT:    2:  (CXXConstructExpr, [B3.3], A)
// CHECK-NEXT:    3: A a;
// CHECK-NEXT:    Preds (3): B4 B5(Unreachable) B6
// CHECK-NEXT:    Succs (1): B5
//
// CHECK:       [B4]
// CHECK-NEXT:    1: CFGScopeEnd(a)
// CHECK-NEXT:    T: goto label;
// CHECK-NEXT:    Succs (1): B3
//
// CHECK:       [B5 (NORETURN)]
// CHECK-NEXT:    1: [B3.3].~A() (Implicit destructor)
// CHECK-NEXT:    2: [B3.3] (Lifetime ends)
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B0
//
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (3): B1 B2 B5
void test_backward_goto() {
label:
  {
    A a;
    goto label;
  }
  foo();
}
