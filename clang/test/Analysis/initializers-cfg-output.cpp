// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=debug.DumpCFG %s 2>&1 | FileCheck %s

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
// CHECK:    1:  (CXXConstructExpr, class A)
// CHECK:    2: A([B1.1]) (Base initializer)
// CHECK:    3:  (CXXConstructExpr, class C)
// CHECK:    4: C([B1.3]) (Base initializer)
// CHECK:    5:  (CXXConstructExpr, class B)
// CHECK:    6: B([B1.5]) (Base initializer)
// CHECK:    7:  (CXXConstructExpr, class A)
// CHECK:    8: A([B1.7]) (Base initializer)
// CHECK:    9: /*implicit*/(int)0
// CHECK:   10: i([B1.9]) (Member initializer)
// CHECK:   11: this
// CHECK:   12: [B1.11]->i
// CHECK:   13: r([B1.12]) (Member initializer)
// CHECK:   14:  (CXXConstructExpr, class A)
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
// CHECK:    3: [B1.1], [B1.2] (CXXConstructExpr, class TestDelegating)
// CHECK:    4: TestDelegating([B1.3]) (Delegating initializer)
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
