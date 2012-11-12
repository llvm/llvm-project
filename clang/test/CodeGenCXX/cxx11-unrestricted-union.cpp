// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - | FileCheck %s

struct A {
  A(); A(const A&); A(A&&); A &operator=(const A&); A &operator=(A&&); ~A();
};
struct B {
  B(); B(const B&); B(B&&); B &operator=(const B&); B &operator=(B&&); ~B();
};

union U {
  U();
  U(const U &);
  U(U &&);
  U &operator=(const U&);
  U &operator=(U&&);
  ~U();

  A a;
  int n;
};

// CHECK-NOT: _ZN1A
U::U() {}
U::U(const U&) {}
U::U(U&&) {}
U &U::operator=(const U&) { return *this; }
U &U::operator=(U &&) { return *this; }
U::~U() {}

struct S {
  S();
  S(const S &);
  S(S &&);
  S &operator=(const S&);
  S &operator=(S&&);
  ~S();

  union {
    A a;
    int n;
  };
  B b;
  int m;
};

// CHECK: _ZN1SC2Ev
// CHECK-NOT: _ZN1A
// CHECK: _ZN1BC1Ev
S::S() {}

// CHECK-NOT: _ZN1A

// CHECK: _ZN1SC2ERKS_
// CHECK-NOT: _ZN1A
// CHECK: _ZN1BC1Ev
S::S(const S&) {}

// CHECK-NOT: _ZN1A

// CHECK: _ZN1SC2EOS_
// CHECK-NOT: _ZN1A
// CHECK: _ZN1BC1Ev
S::S(S&&) {}

// CHECK-NOT: _ZN1A
// CHECK-NOT: _ZN1B
S &S::operator=(const S&) { return *this; }

S &S::operator=(S &&) { return *this; }

// CHECK: _ZN1SD2Ev
// CHECK-NOT: _ZN1A
// CHECK: _ZN1BD1Ev
S::~S() {}

// CHECK-NOT: _ZN1A
