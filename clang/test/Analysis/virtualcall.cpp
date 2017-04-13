// RUN: %clang_analyze_cc1 -analyzer-checker=optin.cplusplus.VirtualCall -analyzer-store region -verify -std=c++11 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=optin.cplusplus.VirtualCall -analyzer-store region -analyzer-config optin.cplusplus.VirtualCall:Interprocedural=true -DINTERPROCEDURAL=1 -verify -std=c++11 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=optin.cplusplus.VirtualCall -analyzer-store region -analyzer-config optin.cplusplus.VirtualCall:PureOnly=true -DPUREONLY=1 -verify -std=c++11 %s

/* When INTERPROCEDURAL is set, we expect diagnostics in all functions reachable
   from a constructor or destructor. If it is not set, we expect diagnostics
   only in the constructor or destructor.

   When PUREONLY is set, we expect diagnostics only for calls to pure virtual
   functions not to non-pure virtual functions.
*/

class A {
public:
  A();
  A(int i);

  ~A() {};
  
  virtual int foo() = 0; // from Sema: expected-note {{'foo' declared here}}
  virtual void bar() = 0;
  void f() {
    foo();
#if INTERPROCEDURAL
        // expected-warning-re@-2 {{{{^}}Call Path : foo <-- fCall to pure virtual function during construction has undefined behavior}}
#endif
  }
};

class B : public A {
public:
  B() {
    foo();
#if !PUREONLY
#if INTERPROCEDURAL
        // expected-warning-re@-3 {{{{^}}Call Path : fooCall to virtual function during construction will not dispatch to derived class}}
#else
        // expected-warning-re@-5 {{{{^}}Call to virtual function during construction will not dispatch to derived class}}
#endif
#endif

  }
  ~B();
  
  virtual int foo();
  virtual void bar() { foo(); }
#if INTERPROCEDURAL
      // expected-warning-re@-2 {{{{^}}Call Path : foo <-- barCall to virtual function during destruction will not dispatch to derived class}}
#endif
};

A::A() {
  f();
}

A::A(int i) {
  foo(); // From Sema: expected-warning {{call to pure virtual member function 'foo' has undefined behavior}}
#if INTERPROCEDURAL
      // expected-warning-re@-2 {{{{^}}Call Path : fooCall to pure virtual function during construction has undefined behavior}}
#else
      // expected-warning-re@-4 {{{{^}}Call to pure virtual function during construction has undefined behavior}}
#endif
}

B::~B() {
  this->B::foo(); // no-warning
  this->B::bar();
  this->foo();
#if !PUREONLY
#if INTERPROCEDURAL
      // expected-warning-re@-3 {{{{^}}Call Path : fooCall to virtual function during destruction will not dispatch to derived class}}
#else
      // expected-warning-re@-5 {{{{^}}Call to virtual function during destruction will not dispatch to derived class}}
#endif
#endif

}

class C : public B {
public:
  C();
  ~C();
  
  virtual int foo();
  void f(int i);
};

C::C() {
  f(foo());
#if !PUREONLY
#if INTERPROCEDURAL
      // expected-warning-re@-3 {{{{^}}Call Path : fooCall to virtual function during construction will not dispatch to derived class}}
#else
      // expected-warning-re@-5 {{{{^}}Call to virtual function during construction will not dispatch to derived class}}
#endif
#endif
}

class D : public B {
public:
  D() {
    foo(); // no-warning
  }
  ~D() { bar(); }
  int foo() final;
  void bar() final { foo(); } // no-warning
};

class E final : public B {
public:
  E() {
    foo(); // no-warning
  }
  ~E() { bar(); }
  int foo() override;
};

// Regression test: don't crash when there's no direct callee.
class F {
public:
  F() {
    void (F::* ptr)() = &F::foo;
    (this->*ptr)();
  }
  void foo();
};

int main() {
  A *a;
  B *b;
  C *c;
  D *d;
  E *e;
  F *f;
}

#include "virtualcall.h"

#define AS_SYSTEM
#include "virtualcall.h"
#undef AS_SYSTEM
