// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCFG %s 2>&1 | FileCheck %s
// XPASS: *

class A {
public:
  ~A() {}
};

class B : public virtual A {
public:
  ~B() {}
};

class C : public virtual A {
public:
  ~C() {}
};

class TestOrder : public C, public B, public virtual A {
  A a;
  int i;
  A *p;
public:
  ~TestOrder();
};

TestOrder::~TestOrder() {}

class TestArray {
  A a[2];
  A b[0];
public:
  ~TestArray();
};

TestArray::~TestArray() {}

// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// CHECK:    1: this->a.~A() (Member object destructor)
// CHECK:    2: ~B() (Base object destructor)
// CHECK:    3: ~C() (Base object destructor)
// CHECK:    4: ~A() (Base object destructor)
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
// CHECK:  [B2 (ENTRY)]
// CHECK:    Succs (1): B1
// CHECK:  [B1]
// CHECK:    1: this->a.~A() (Member object destructor)
// CHECK:    Preds (1): B2
// CHECK:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK:    Preds (1): B1
