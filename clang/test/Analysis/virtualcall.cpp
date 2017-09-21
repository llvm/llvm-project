// RUN: %clang_analyze_cc1 -analyzer-checker=optin.cplusplus.VirtualCall -analyzer-store region -analyzer-output=text -verify -std=c++11 %s

// RUN: %clang_analyze_cc1 -analyzer-checker=optin.cplusplus.VirtualCall -analyzer-store region -analyzer-config optin.cplusplus.VirtualCall:PureOnly=true -DPUREONLY=1 -analyzer-output=text -verify -std=c++11 %s

#include "virtualcall.h"

class A {
public:
  A();

  ~A(){};

  virtual int foo() = 0;
  virtual void bar() = 0;
  void f() {
    foo();
	// expected-warning-re@-1 {{{{^}}Call to pure virtual function during construction}}
	// expected-note-re@-2 {{{{^}}Call to pure virtual function during construction}}
  }
};

class B : public A {
public:
  B() { // expected-note {{Calling default constructor for 'A'}}
    foo(); 
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
	// expected-note-re@-3 {{{{^}}This constructor of an object of type 'B' has not returned when the virtual method was called}}
  	// expected-note-re@-4 {{{{^}}Call to virtual function during construction}}
#endif
  }
  ~B();

  virtual int foo();
  virtual void bar() {
    foo(); 
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during destruction}}
  	// expected-note-re@-3 {{{{^}}Call to virtual function during destruction}}
#endif
  } 
};

A::A() { 
  f(); 
// expected-note-re@-1 {{{{^}}This constructor of an object of type 'A' has not returned when the virtual method was called}}
// expected-note-re@-2 {{{{^}}Calling 'A::f'}}
}

B::~B() {
  this->B::foo(); // no-warning
  this->B::bar();
#if !PUREONLY
 	 // expected-note-re@-2 {{{{^}}This destructor of an object of type '~B' has not returned when the virtual method was called}}
 	 // expected-note-re@-3 {{{{^}}Calling 'B::bar'}}
#endif
  this->foo(); 
#if !PUREONLY
 	 // expected-warning-re@-2 {{{{^}}Call to virtual function during destruction}}
 	 // expected-note-re@-3 {{{{^}}This destructor of an object of type '~B' has not returned when the virtual method was called}}
 	 // expected-note-re@-4 {{{{^}}Call to virtual function during destruction}}
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
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
	// expected-note-re@-3 {{{{^}}This constructor of an object of type 'C' has not returned when the virtual method was called}}
  	// expected-note-re@-4 {{{{^}}Call to virtual function during construction}}
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
#if !PUREONLY
 	 // expected-note-re@-2 2{{{{^}}Calling '~B'}}
#endif
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
    f();     
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
	// expected-note-re@-3 {{{{^}}This constructor of an object of type 'H' has not returned when the virtual method was called}}
  	// expected-note-re@-4 {{{{^}}Call to virtual function during construction}}
#endif
    H &h = *this;
    h.f(); 
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
  	// expected-note-re@-3 {{{{^}}This constructor of an object of type 'H' has not returned when the virtual method was called}}
  	// expected-note-re@-4 {{{{^}}Call to virtual function during construction}}
#endif
  }
};

class X {
public:
  X() {
    g(); 
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
	// expected-note-re@-3 {{{{^}}This constructor of an object of type 'X' has not returned when the virtual method was called}}
  	// expected-note-re@-4 {{{{^}}Call to virtual function during construction}}
#endif
  }
  X(int i) {
    if (i > 0) {
#if !PUREONLY
	// expected-note-re@-2 {{{{^}}Taking true branch}}
	// expected-note-re@-3 {{{{^}}Taking false branch}}
#endif
      X x(i - 1);
#if !PUREONLY
	// expected-note-re@-2 {{{{^}}Calling constructor for 'X'}}
#endif
      x.g(); // no warning
    }
    g(); 
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
	// expected-note-re@-3 {{{{^}}This constructor of an object of type 'X' has not returned when the virtual method was called}}
  	// expected-note-re@-4 {{{{^}}Call to virtual function during construction}}
#endif
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
#if !PUREONLY
  	// expected-note-re@-2 {{{{^}}This constructor of an object of type 'M' has not returned when the virtual method was called}}
	// expected-note-re@-3 {{{{^}}Calling 'N::callFooOfM'}}
#endif
  }
  virtual void foo();
};
void N::callFooOfM(M *m) {
  m->foo(); 
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
  	// expected-note-re@-3 {{{{^}}Call to virtual function during construction}}
#endif
}

class Y {
public:
  virtual void foobar();
  void fooY() {
    F f1;
    foobar(); 
#if !PUREONLY
  	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
  	// expected-note-re@-3 {{{{^}}Call to virtual function during construction}}
#endif
  }
  Y() { fooY(); }
#if !PUREONLY
  	// expected-note-re@-2 {{{{^}}This constructor of an object of type 'Y' has not returned when the virtual method was called}}
  	// expected-note-re@-3 {{{{^}}Calling 'Y::fooY'}}
#endif
};

int main() {
  B b;
#if PUREONLY
	//expected-note-re@-2 {{{{^}}Calling default constructor for 'B'}}
#else 
	//expected-note-re@-4 2{{{{^}}Calling default constructor for 'B'}}
#endif
  C c;
#if !PUREONLY
	//expected-note-re@-2 {{{{^}}Calling default constructor for 'C'}}
#endif
  D d;
  E e;
  F f;
  G g;
  H h;
  H h1(1);
#if !PUREONLY
	//expected-note-re@-2 {{{{^}}Calling constructor for 'H'}}
	//expected-note-re@-3 {{{{^}}Calling constructor for 'H'}}
#endif
  X x; 
#if !PUREONLY
	//expected-note-re@-2 {{{{^}}Calling default constructor for 'X'}}
#endif
  X x1(1);
#if !PUREONLY
	//expected-note-re@-2 {{{{^}}Calling constructor for 'X'}}
#endif
  M m;
#if !PUREONLY
	//expected-note-re@-2 {{{{^}}Calling default constructor for 'M'}}
#endif
  Y *y = new Y;
  delete y;
  header::Z z;
#if !PUREONLY
	// expected-note-re@-2 {{{{^}}Calling default constructor for 'Z'}}
#endif
}
#if !PUREONLY
	//expected-note-re@-2 2{{{{^}}Calling '~E'}}
#endif

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
