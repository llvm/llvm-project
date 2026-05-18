// RUN: %clang_cc1 -fblocks -Wno-unused-value -std=c++20 -ast-dump -verify %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -fblocks -Wno-unused-value -std=c++20 -ast-dump -verify %s -ast-dump -fexperimental-new-constant-interpreter | FileCheck %s

// expected-no-diagnostics

struct P {
  consteval P() {}
};

struct A {
  A(int v) { this->data = new int(v); }
  const int& get() const {
    return *this->data;
  }
  ~A() { delete data; }
private:
  int *data;
};

void foo() {
  for (;A(1), P(), false;);
  // CHECK: foo
  // CHECK: ExprWithCleanups
  // CHECK-NEXT: BinaryOperator {{.*}} 'bool' ','
  // CHECK-NEXT: BinaryOperator {{.*}} 'P' ','
  // CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'A'
  // CHECK-NEXT: CXXBindTemporaryExpr {{.*}} 'A'
  // CHECK-NEXT: CXXConstructExpr {{.*}} 'A'
  // CHECK: ConstantExpr {{.*}} 'P'
  // CHECK-NEXT: value:
  // CHECK-NEXT: ExprWithCleanups
}

void foobar() {
  A a(1);
  for (; ^{ auto ptr = &a.get(); }(), P(), false;);
  // CHECK: ExprWithCleanups
  // CHECK-NEXT: cleanup Block
  // CHECK-NEXT: BinaryOperator {{.*}} 'bool' ','
  // CHECK-NEXT: BinaryOperator {{.*}} 'P' ','
  // CHECK-NEXT: CallExpr
  // CHECK-NEXT: BlockExpr
  // CHECK: ConstantExpr {{.*}} 'P'
  // CHECK-NEXT: value:
  // CHECK-NEXT: ExprWithCleanups
  // CHECK-NOT:  cleanup Block
}

struct B {
  int *p = new int(38);
  consteval int get() { return *p; }
  constexpr ~B() { delete p; }
};

void bar() {
  // CHECK: bar
  // CHECK: ExprWithCleanups
  // CHECK: ConstantExpr
  // CHECK-NEXT: value:
  // CHECK-NEXT: ExprWithCleanups
  int k = B().get();
}
