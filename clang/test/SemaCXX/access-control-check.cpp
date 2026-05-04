// RUN: %clang_cc1 -fsyntax-only -verify %s

class M {
  int iM;
};

class P {
  int iP; // expected-note {{declared private here}}
  int PPR(); // expected-note {{declared private here}}
};

class N : M,P {
  N() {}
  int PR() { return iP + PPR(); } // expected-error 2 {{private member of 'P'}}
};

namespace GH83608 {

class single;

class check_constructible {
  // This makes the class a non-aggregate, which enforces us to check
  // the constructor when initializing.
  check_constructible() {}

  friend class single;
};

struct single {
  template <class T> single(T u, check_constructible = {}) {}
};

// We perform access checking when substituting into the default argument.
// Make sure it runs within the context of 'single'.
single x(0);

}

namespace GH62444 {

struct B {
  friend struct A;
private:
  B(int); // #B
};

template<class T>
int f(T = 0); // #Decl

struct A {
  A() {
    int i = f<B>();
    // expected-error@#Decl {{calling a private constructor}}
    // expected-note@-2 {{in instantiation of default function argument}}
    // expected-note@#B {{declared private}}
  }
};

int i = f<B>();

}
