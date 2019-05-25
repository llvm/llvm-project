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
  // CHECK:       A()
  // CHECK:        [B1 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (1): B1
  A() {}

  // CHECK:       A(int i)
  // CHECK:        [B1 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (1): B1
  A(int i) {}
};

class B : public virtual A {
public:
  // CHECK:       B()
  // CHECK:        [B3 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B2
  // CHECK:        [B1]
  // WARNINGS-NEXT:     1:  (CXXConstructExpr, class A)
  // ANALYZER-NEXT:     1:  (CXXConstructExpr, A() (Base initializer), class A)
  // CHECK-NEXT:     2: A([B1.1]) (Base initializer)
  // CHECK-NEXT:     Preds (1): B2
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B2]
  // CHECK-NEXT:     T: (See if most derived ctor has already initialized vbases)
  // CHECK-NEXT:     Preds (1): B3
  // CHECK-NEXT:     Succs (2): B0 B1
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (2): B1 B2
  B() {}

  // CHECK:       B(int i)
  // CHECK:        [B3 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B2
  // CHECK:        [B1]
  // CHECK-NEXT:     1: i
  // CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
  // WARNINGS-NEXT:     3: [B1.2] (CXXConstructExpr, class A)
  // ANALYZER-NEXT:     3: [B1.2] (CXXConstructExpr, A([B1.2]) (Base initializer), class A)
  // CHECK-NEXT:     4: A([B1.3]) (Base initializer)
  // CHECK-NEXT:     Preds (1): B2
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B2]
  // CHECK-NEXT:     T: (See if most derived ctor has already initialized vbases)
  // CHECK-NEXT:     Preds (1): B3
  // CHECK-NEXT:     Succs (2): B0 B1
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (2): B1 B2
  B(int i) : A(i) {}
};

class C : public virtual A {
public:
  // CHECK:       C()
  // CHECK:        [B3 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B2
  // CHECK:        [B1]
  // WARNINGS-NEXT:     1:  (CXXConstructExpr, class A)
  // ANALYZER-NEXT:     1:  (CXXConstructExpr, A() (Base initializer), class A)
  // CHECK-NEXT:     2: A([B1.1]) (Base initializer)
  // CHECK-NEXT:     Preds (1): B2
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B2]
  // CHECK-NEXT:     T: (See if most derived ctor has already initialized vbases)
  // CHECK-NEXT:     Preds (1): B3
  // CHECK-NEXT:     Succs (2): B0 B1
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (2): B1 B2
  C() {}

  // CHECK:       C(int i)
  // CHECK:        [B3 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B2
  // CHECK:        [B1]
  // CHECK-NEXT:     1: i
  // CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
  // WARNINGS-NEXT:     3: [B1.2] (CXXConstructExpr, class A)
  // ANALYZER-NEXT:     3: [B1.2] (CXXConstructExpr, A([B1.2]) (Base initializer), class A)
  // CHECK-NEXT:     4: A([B1.3]) (Base initializer)
  // CHECK-NEXT:     Preds (1): B2
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B2]
  // CHECK-NEXT:     T: (See if most derived ctor has already initialized vbases)
  // CHECK-NEXT:     Preds (1): B3
  // CHECK-NEXT:     Succs (2): B0 B1
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (2): B1 B2
  C(int i) : A(i) {}
};


class TestOrder : public C, public B, public A {
  int i;
  int& r;
public:
  TestOrder();
};

// CHECK:       TestOrder::TestOrder()
// CHECK:        [B4 (ENTRY)]
// CHECK-NEXT:     Succs (1): B3
// CHECK:        [B1]
// WARNINGS-NEXT:     1:  (CXXConstructExpr, class C)
// ANALYZER-NEXT:     1:  (CXXConstructExpr, C() (Base initializer), class C)
// CHECK-NEXT:     2: C([B1.1]) (Base initializer)
// WARNINGS-NEXT:     3:  (CXXConstructExpr, class B)
// ANALYZER-NEXT:     3:  (CXXConstructExpr, B() (Base initializer), class B)
// CHECK-NEXT:     4: B([B1.3]) (Base initializer)
// WARNINGS-NEXT:     5:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:     5:  (CXXConstructExpr, A() (Base initializer), class A)
// CHECK-NEXT:     6: A([B1.5]) (Base initializer)
// CHECK-NEXT:     7: /*implicit*/(int)0
// CHECK-NEXT:     8: i([B1.7]) (Member initializer)
// CHECK-NEXT:     9: this
// CHECK-NEXT:    10: [B1.9]->i
// CHECK-NEXT:    11: r([B1.10]) (Member initializer)
// WARNINGS-NEXT:    12:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:    12:  (CXXConstructExpr, [B1.13], class A)
// CHECK-NEXT:    13: A a;
// CHECK-NEXT:     Preds (2): B2 B3
// CHECK-NEXT:     Succs (1): B0
// CHECK:        [B2]
// WARNINGS-NEXT:     1:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:     1:  (CXXConstructExpr, A() (Base initializer), class A)
// CHECK-NEXT:     2: A([B2.1]) (Base initializer)
// CHECK-NEXT:     Preds (1): B3
// CHECK-NEXT:     Succs (1): B1
// CHECK:        [B3]
// CHECK-NEXT:     T: (See if most derived ctor has already initialized vbases)
// CHECK-NEXT:     Preds (1): B4
// CHECK-NEXT:     Succs (2): B1 B2
// CHECK:        [B0 (EXIT)]
// CHECK-NEXT:     Preds (1): B1
TestOrder::TestOrder()
  : r(i), B(), i(), C() {
  A a;
}

class TestControlFlow {
  int x, y, z;
public:
  TestControlFlow(bool b);
};

// CHECK:       TestControlFlow::TestControlFlow(bool b)
// CHECK:        [B5 (ENTRY)]
// CHECK-NEXT:     Succs (1): B4
// CHECK:        [B1]
// CHECK-NEXT:     1: [B4.4] ? [B2.1] : [B3.1]
// CHECK-NEXT:     2: y([B1.1]) (Member initializer)
// CHECK-NEXT:     3: this
// CHECK-NEXT:     4: [B1.3]->y
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:     6: z([B1.5]) (Member initializer)
// CHECK-NEXT:     7: int v;
// CHECK-NEXT:     Preds (2): B2 B3
// CHECK-NEXT:     Succs (1): B0
// CHECK:        [B2]
// CHECK-NEXT:     1: 0
// CHECK-NEXT:     Preds (1): B4
// CHECK-NEXT:     Succs (1): B1
// CHECK:        [B3]
// CHECK-NEXT:     1: 1
// CHECK-NEXT:     Preds (1): B4
// CHECK-NEXT:     Succs (1): B1
// CHECK:        [B4]
// CHECK-NEXT:     1: 0
// CHECK-NEXT:     2: x([B4.1]) (Member initializer)
// CHECK-NEXT:     3: b
// CHECK-NEXT:     4: [B4.3] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:     T: [B4.4] ? ... : ...
// CHECK-NEXT:     Preds (1): B5
// CHECK-NEXT:     Succs (2): B2 B3
// CHECK:        [B0 (EXIT)]
// CHECK-NEXT:     Preds (1): B1
TestControlFlow::TestControlFlow(bool b)
  : y(b ? 0 : 1)
  , x(0)
  , z(y) {
  int v;
}

class TestDelegating {
  int x, z;
public:

  // CHECK:       TestDelegating()
  // CHECK:        [B2 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B1
  // CHECK:        [B1]
  // CHECK-NEXT:     1: 2
  // CHECK-NEXT:     2: 3
  // WARNINGS-NEXT:     3: [B1.1], [B1.2] (CXXConstructExpr, class TestDelegating)
  // ANALYZER-NEXT:     3: [B1.1], [B1.2] (CXXConstructExpr, TestDelegating([B1.1], [B1.2]) (Delegating initializer), class TestDelegating)
  // CHECK-NEXT:     4: TestDelegating([B1.3]) (Delegating initializer)
  // CHECK-NEXT:     Preds (1): B2
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (1): B1
  TestDelegating() : TestDelegating(2, 3) {}

  // CHECK:       TestDelegating(int x, int z)
  // CHECK:        [B2 (ENTRY)]
  // CHECK-NEXT:     Succs (1): B1
  // CHECK:        [B1]
  // CHECK-NEXT:     1: x
  // CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
  // CHECK-NEXT:     3: x([B1.2]) (Member initializer)
  // CHECK-NEXT:     4: z
  // CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, LValueToRValue, int)
  // CHECK-NEXT:     6: z([B1.5]) (Member initializer)
  // CHECK-NEXT:     Preds (1): B2
  // CHECK-NEXT:     Succs (1): B0
  // CHECK:        [B0 (EXIT)]
  // CHECK-NEXT:     Preds (1): B1
  TestDelegating(int x, int z) : x(x), z(z) {}
};

class TestMoreControlFlow : public virtual A {
  A a;

public:
  TestMoreControlFlow(bool coin);
};

// CHECK:       TestMoreControlFlow::TestMoreControlFlow(bool coin)
// CHECK:        [B10 (ENTRY)]
// CHECK-NEXT:     Succs (1): B9
// CHECK:        [B1]
// CHECK-NEXT:     1: [B4.2] ? [B2.1] : [B3.1]
// WARNINGS-NEXT:     2: [B1.1] (CXXConstructExpr, class A)
// ANALYZER-NEXT:     2: [B1.1] (CXXConstructExpr, a([B1.1]) (Member initializer), class A)
// CHECK-NEXT:     3: a([B1.2]) (Member initializer)
// CHECK-NEXT:     Preds (2): B2 B3
// CHECK-NEXT:     Succs (1): B0
// CHECK:        [B2]
// CHECK-NEXT:     1: 3
// CHECK-NEXT:     Preds (1): B4
// CHECK-NEXT:     Succs (1): B1
// CHECK:        [B3]
// CHECK-NEXT:     1: 4
// CHECK-NEXT:     Preds (1): B4
// CHECK-NEXT:     Succs (1): B1
// CHECK:        [B4]
// CHECK-NEXT:     1: coin
// CHECK-NEXT:     2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:     T: [B4.2] ? ... : ...
// CHECK-NEXT:     Preds (2): B5 B9
// CHECK-NEXT:     Succs (2): B2 B3
// CHECK:        [B5]
// CHECK-NEXT:     1: [B8.2] ? [B6.1] : [B7.1]
// WARNINGS-NEXT:     2: [B5.1] (CXXConstructExpr, class A)
// ANALYZER-NEXT:     2: [B5.1] (CXXConstructExpr, A([B5.1]) (Base initializer), class A)
// CHECK-NEXT:     3: A([B5.2]) (Base initializer)
// CHECK-NEXT:     Preds (2): B6 B7
// CHECK-NEXT:     Succs (1): B4
// CHECK:        [B6]
// CHECK-NEXT:     1: 1
// CHECK-NEXT:     Preds (1): B8
// CHECK-NEXT:     Succs (1): B5
// CHECK:        [B7]
// CHECK-NEXT:     1: 2
// CHECK-NEXT:     Preds (1): B8
// CHECK-NEXT:     Succs (1): B5
// CHECK:        [B8]
// CHECK-NEXT:     1: coin
// CHECK-NEXT:     2: [B8.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:     T: [B8.2] ? ... : ...
// CHECK-NEXT:     Preds (1): B9
// CHECK-NEXT:     Succs (2): B6 B7
// CHECK:        [B9]
// CHECK-NEXT:     T: (See if most derived ctor has already initialized vbases)
// CHECK-NEXT:     Preds (1): B10
// CHECK-NEXT:     Succs (2): B4 B8
// CHECK:        [B0 (EXIT)]
// CHECK-NEXT:     Preds (1): B1
TestMoreControlFlow::TestMoreControlFlow(bool coin)
    : A(coin ? 1 : 2), a(coin ? 3 : 4) {}
