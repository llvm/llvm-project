// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S {
   S (S);  // expected-error {{copy constructor must pass its first argument by reference}}
};

S f();

void g() { 
  S a( f() );
}

class foo {
  foo(foo&, int); // expected-note {{previous}}
  foo(int); // expected-note {{previous}}
  foo(const foo&); // expected-note {{previous}}
};

foo::foo(foo&, int = 0) { } // expected-error {{makes this constructor a copy constructor}}
foo::foo(int = 0) { } // expected-error {{makes this constructor a default constructor}}
foo::foo(const foo& = 0) { } //expected-error {{makes this constructor a default constructor}}

namespace PR6064 {
  struct A {
    A() { }
    inline A(A&, int); // expected-note {{previous}}
  };

  A::A(A&, int = 0) { } // expected-error {{makes this constructor a copy constructor}}

  void f() {
    A const a;
    A b(a);
  }
}

namespace PR10618 {
  struct A {
    A(int, int, int); // expected-note {{previous}}
  };
  A::A(int a = 0, // expected-error {{makes this constructor a default constructor}}
       int b = 0,
       int c = 0) {}

  struct B {
    B(int);
    B(const B&, int); // expected-note {{previous}}
  };
  B::B(const B& = B(0), // expected-error {{makes this constructor a default constructor}}
       int = 0) {
  }

  struct C {
    C(const C&, int); // expected-note {{previous}}
  };
  C::C(const C&,
       int = 0) { // expected-error {{makes this constructor a copy constructor}}
  }
}
