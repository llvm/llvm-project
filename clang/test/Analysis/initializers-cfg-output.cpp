// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=false %s 2>&1 | FileCheck -check-prefixes=CHECK,WARNINGS %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=true %s 2>&1 | FileCheck -check-prefixes=CHECK,ANALYZER %s

// This file tests how we construct two different flavors of the Clang CFG -
// the CFG used by the Sema analysis-based warnings and the CFG used by the
// static analyzer. The difference in the behavior is checked via FileCheck
// prefixes (WARNINGS and ANALYZER respectively). When introducing new analyzer
// flags, no new run lines should be added - just these flags would go to the
// respective line depending on where is it turned on and where is it turned
// off. Feel free to add tests that test only one of the CFG flavors if you're
// not sure how the other flavor is supposed to work in your case.

class A {
public:
  A() {}
  A(int i) {}
};

class B : public virtual A {
public:
  B() {}
  B(int i) : A(i) {}
};

class C : public virtual A {
public:
  C() {}
  C(int i) : A(i) {}
};

class TestOrder : public C, public B, public A {
  int i;
  int& r;
public:
  TestOrder();
};

TestOrder::TestOrder()
  : r(i), B(), i(), C() {
  A a;
}

class TestControlFlow {
  int x, y, z;
public:
  TestControlFlow(bool b);
};

TestControlFlow::TestControlFlow(bool b)
  : y(b ? 0 : 1)
  , x(0)
  , z(y) {
  int v;
}

class TestDelegating {
  int x, z;
 public:
  TestDelegating() : TestDelegating(2, 3) {}
  TestDelegating(int x, int z) : x(x), z(z) {}
};

// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// WARNINGS:    1:  (CXXConstructExpr, class A)
// ANALYZER:    1:  (CXXConstructExpr, A() (Base initializer), class A)
// CHECK:    2: A([B1.1]) (Base initializer)
// WARNINGS:    3:  (CXXConstructExpr, class C)
// ANALYZER:    3:  (CXXConstructExpr, C() (Base initializer), class C)
// CHECK:    4: C([B1.3]) (Base initializer)
// WARNINGS:    5:  (CXXConstructExpr, class B)
// ANALYZER:    5:  (CXXConstructExpr, B() (Base initializer), class B)
// CHECK:    6: B([B1.5]) (Base initializer)
// WARNINGS:    7:  (CXXConstructExpr, class A)
// ANALYZER:    7:  (CXXConstructExpr, A() (Base initializer), class A)
// CHECK:    8: A([B1.7]) (Base initializer)
// CHECK:    9: /*implicit*/(int)0
// CHECK:   10: i([B1.9]) (Member initializer)
// CHECK:   11: this
// CHECK:   12: [B1.11]->i
// CHECK:   13: r([B1.12]) (Member initializer)
// WARNINGS:   14:  (CXXConstructExpr, class A)
// ANALYZER:   14:  (CXXConstructExpr, [B1.15], class A)
// CHECK:   15: A a;
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B5 (ENTRY)]
// CHECK:    Succs (1): B4
// CHECK:  [B1]
// CHECK:    1: [B4.4] ? [B2.1] : [B3.1]
// CHECK:    2: y([B1.1]) (Member initializer)
// CHECK:    3: this
// CHECK:    4: [B1.3]->y
// CHECK:    5: [B1.4] (ImplicitCastExpr, LValueToRValue, int)
// CHECK:    6: z([B1.5]) (Member initializer)
// CHECK:    7: int v;
// CHECK:    Preds (2): B2 B3
// CHECK:    Succs (1): B0
// CHECK:  [B2]
// CHECK:    1: 0
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B3]
// CHECK:    1: 1
// CHECK:    Preds (1): B4
// CHECK:    Succs (1): B1
// CHECK:  [B4]
// CHECK:    1: 0
// CHECK:    2: x([B4.1]) (Member initializer)
// CHECK:    3: b
// CHECK:    4: [B4.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK:    T: [B4.4] ? ... : ...
// CHECK:    Preds (1): B5
// CHECK:    Succs (2): B2 B3
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// CHECK:    1: 2
// CHECK:    2: 3
// WARNINGS:    3: [B1.1], [B1.2] (CXXConstructExpr, class TestDelegating)
// ANALYZER:    3: [B1.1], [B1.2] (CXXConstructExpr, TestDelegating([B1.1], [B1.2]) (Delegating initializer), class TestDelegating)
// CHECK:    4: TestDelegating([B1.3]) (Delegating initializer)
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
