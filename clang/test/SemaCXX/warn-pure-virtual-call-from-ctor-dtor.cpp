// RUN: %clang_cc1 %s -fsyntax-only -verify -Wcall-to-pure-virtual-from-ctor-dtor
struct A {
  A() { f(); } // expected-warning {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the constructor of 'A'}}
  ~A() { f(); } // expected-warning {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the destructor of 'A'}}

  virtual void f() = 0; // expected-note 2 {{'f' declared here}}
};

// Don't warn (or note) when calling the function on a pointer. (PR10195)
struct B {
  A *a;
  B() { a->f(); };
  ~B() { a->f(); };
};

// Don't warn if the call is fully qualified. (PR23215)
struct C {
    virtual void f() = 0;
    C() {
        C::f();
    }
};
