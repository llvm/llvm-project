// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.VirtualCall \
// RUN:                    -analyzer-checker=debug.ExprInspection \
// RUN:                    -std=c++11 -verify=impure %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.PureVirtualCall \
// RUN:                    -analyzer-checker=debug.ExprInspection \
// RUN:                    -std=c++11 -verify=pure -std=c++11 %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.VirtualCall \
// RUN:                    -analyzer-config \
// RUN:                        optin.cplusplus.VirtualCall:PureOnly=true \
// RUN:                    -analyzer-checker=debug.ExprInspection \
// RUN:                    -std=c++11 -verify=none %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.PureVirtualCall \
// RUN:                    -analyzer-checker=optin.cplusplus.VirtualCall \
// RUN:                    -analyzer-checker=debug.ExprInspection \
// RUN:                    -std=c++11 -verify=pure,impure -std=c++11 %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.PureVirtualCall \
// RUN:                    -analyzer-checker=optin.cplusplus.VirtualCall \
// RUN:                    -analyzer-config \
// RUN:                        optin.cplusplus.VirtualCall:PureOnly=true \
// RUN:                    -analyzer-checker=debug.ExprInspection \
// RUN:                    -std=c++11 -verify=pure %s


// We expect no diagnostics when all checks are disabled.
// none-no-diagnostics


#include "virtualcall.h"

void clang_analyzer_warnIfReached();

class A {
public:
  A();

  ~A(){};

  virtual int foo() = 0;
  virtual void bar() = 0;
  void f() {
    foo(); // pure-warning{{Call to pure virtual method 'A::foo' during construction has undefined behavior}}
    clang_analyzer_warnIfReached(); // no-warning
  }
};

A::A() {
  f();
}

class B {
public:
  B() {
    foo(); // impure-warning {{Call to virtual method 'B::foo' during construction bypasses virtual dispatch}}
  }
  ~B();

  virtual int foo();
  virtual void bar() {
    foo(); // impure-warning {{Call to virtual method 'B::foo' during destruction bypasses virtual dispatch}}
  }
};

B::~B() {
  this->B::foo(); // no-warning
  this->B::bar();
  this->foo(); // impure-warning {{Call to virtual method 'B::foo' during destruction bypasses virtual dispatch}}
}

class C : public B {
public:
  C();
  ~C();

  virtual int foo();
  void f(int i);
};

C::C() {
  f(foo()); // impure-warning {{Call to virtual method 'C::foo' during construction bypasses virtual dispatch}}
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

class F {
public:
  F() {
    void (F::*ptr)() = &F::foo;
    (this->*ptr)();
  }
  void foo();
};

class G {
public:
  G() {}
  virtual void bar();
  void foo() {
    bar(); // no warning
  }
};

class H {
public:
  H() : initState(0) { init(); }
  int initState;
  virtual void f() const;
  void init() {
    if (initState)
      f(); // no warning
  }

  H(int i) {
    G g;
    g.foo();
    g.bar(); // no warning
    f(); // impure-warning {{Call to virtual method 'H::f' during construction bypasses virtual dispatch}}
    H &h = *this;
    h.f(); // impure-warning {{Call to virtual method 'H::f' during construction bypasses virtual dispatch}}
  }
};

class X {
public:
  X() {
    g(); // impure-warning {{Call to virtual method 'X::g' during construction bypasses virtual dispatch}}
  }
  X(int i) {
    if (i > 0) {
      X x(i - 1);
      x.g(); // no warning
    }
    g(); // impure-warning {{Call to virtual method 'X::g' during construction bypasses virtual dispatch}}
  }
  virtual void g();
};

class M;
class N {
public:
  virtual void virtualMethod();
  void callFooOfM(M *);
};
class M {
public:
  M() {
    N n;
    n.virtualMethod(); // no warning
    n.callFooOfM(this);
  }
  virtual void foo();
};
void N::callFooOfM(M *m) {
  m->foo(); // impure-warning {{Call to virtual method 'M::foo' during construction bypasses virtual dispatch}}
}

class Y {
public:
  virtual void foobar();
  void fooY() {
    F f1;
    foobar(); // impure-warning {{Call to virtual method 'Y::foobar' during construction bypasses virtual dispatch}}
  }
  Y() { fooY(); }
};

int main() {
  B b;
  C c;
  D d;
  E e;
  F f;
  G g;
  H h;
  H h1(1);
  X x; 
  X x1(1);
  M m;
  Y *y = new Y;
  delete y;
  header::Z z;
}

namespace PR34451 {
struct a {
  void b() {
    a c[1];
    c->b();
  }
};

class e {
 public:
  void b() const;
};

class c {
  void m_fn2() const;
  e d[];
};

void c::m_fn2() const { d->b(); }
}
