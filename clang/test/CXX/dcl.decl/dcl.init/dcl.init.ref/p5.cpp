// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR5909 {
  struct Foo {
    int x : 20;
  };
  
  bool Test(const int& foo);
  
  const Foo f = { 0 };  // It compiles without the 'const'.
  bool z = Test(f.x);
}

namespace PR6264 {
  typedef int (&T)[3];
  struct S
  {
    operator T ();
  };
  void f()
  {
    T bar = S();
  }
}

namespace PR6066 {
  struct B { };
  struct A : B {
    operator B*();
    operator B&(); // expected-warning{{conversion function converting 'PR6066::A' to its base class 'PR6066::B' will never be used}}
  };

  void f(B&); // no rvalues accepted
  void f(B*);

  int g() {
    f(A()); // calls f(B*)
    return 0;
  }
}

namespace test3 {
  struct A {
    unsigned bitX : 4; // expected-note 3 {{bit-field is declared here}}
    unsigned bitY : 4; // expected-note {{bit-field is declared here}}
    unsigned var;

    void foo();
  };

  void test(A *a) {
    unsigned &t0 = a->bitX; // expected-error {{non-const reference cannot bind to bit-field 'bitX'}}
    unsigned &t1 = (unsigned&) a->bitX; // expected-error {{C-style cast from bit-field lvalue to reference type 'unsigned int &'}}
    unsigned &t2 = const_cast<unsigned&>(a->bitX); // expected-error {{const_cast from bit-field lvalue to reference type 'unsigned int &'}}
    unsigned &t3 = (a->foo(), a->bitX); // expected-error {{non-const reference cannot bind to bit-field 'bitX'}}
    unsigned &t4 = (a->var ? a->bitX : a->bitY); // expected-error {{non-const reference cannot bind to bit-field}}
    unsigned &t5 = (a->var ? a->bitX : a->bitX); // expected-error {{non-const reference cannot bind to bit-field}}
    unsigned &t6 = (a->var ? a->bitX : a->var); // expected-error {{non-const reference cannot bind to bit-field}}
    unsigned &t7 = (a->var ? a->var : a->bitY); // expected-error {{non-const reference cannot bind to bit-field}}
    unsigned &t8 = (a->bitX = 3); // expected-error {{non-const reference cannot bind to bit-field 'bitX'}}
    unsigned &t9 = (a->bitY += 3); // expected-error {{non-const reference cannot bind to bit-field 'bitY'}}
  }
}

namespace explicit_ctor {
  struct A {};
  struct B { // expected-note 2{{candidate}}
    explicit B(const A&); // expected-note {{explicit constructor is not a candidate}}
  };
  A a;
  const B &b(a); // expected-error {{no viable conversion}}
}
