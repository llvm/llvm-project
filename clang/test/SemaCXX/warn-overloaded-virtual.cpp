// RUN: %clang_cc1 -fsyntax-only -Woverloaded-virtual -verify %s

struct B1 {
  virtual void foo(int); // expected-note {{declared here}}
  virtual void foo(); // expected-note {{declared here}}
};

struct S1 : public B1 {
  void foo(float); // expected-warning {{hides overloaded virtual functions}}
};

struct S2 : public B1 {
  void foo(); // expected-note {{declared here}}
};

struct B2 {
  virtual void foo(void*); // expected-note {{declared here}}
};

struct MS1 : public S2, public B2 {
   virtual void foo(int); // expected-warning {{hides overloaded virtual functions}}
};

struct B3 {
  virtual void foo(int);
  virtual void foo();
};

struct S3 : public B3 {
  using B3::foo;
  void foo(float);
};

struct B4 {
  virtual void foo();
};

struct S4 : public B4 {
  void foo(float);
  void foo();
};

namespace PR9182 {
struct Base {
  virtual void foo(int);
};

void Base::foo(int) { }

struct Derived : public Base {
  virtual void foo(int);   
  void foo(int, int);   
};
}

namespace PR9396 {
class A {
public:
  virtual void f(int) {}
};

class B : public A {
public:
  static void f() {}
};
}

namespace ThreeLayer {
struct A {
  virtual void f();
};

struct B: A {
  void f();
  void f(int);
};

struct C: B {
  void f(int);
  using A::f;
};
}

namespace UnbalancedVirtual {
struct Base {
  virtual void func();
};

struct Derived1: virtual Base {
  virtual void func();
};

struct Derived2: virtual Base {
};

struct MostDerived: Derived1, Derived2 {
  void func(int);
  void func();
};
}

namespace UnbalancedVirtual2 {
struct Base {
  virtual void func();
};

struct Derived1: virtual Base {
  virtual void func();
};

struct Derived2: virtual Base {
};

struct Derived3: Derived1 {
  virtual void func();
};

struct MostDerived: Derived3, Derived2 {
  void func(int);
  void func();
};
}
