// RUN: %clang_cc1 -fsyntax-only -Wno-unused-variable -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-variable -std=c++20 -verify=expected,since-cxx20 %s

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

namespace GH12361 {
class D1 {
  class E1 {
    class F1 {}; // #F1

    friend D1::E1::F1 foo1();
    friend void foo2(D1::E1::F1);
    friend void foo3() noexcept(sizeof(D1::E1::F1) == 1);
    friend void foo4();
#if __cplusplus >= 202002L
    template <class T>
    friend void foo5(T) requires (sizeof(D1::E1::F1) == 1);
#endif
  };

  D1::E1::F1 friend foo1();     // expected-error{{'F1' is a private member of 'GH12361::D1::E1'}}
  // expected-note@#F1 {{implicitly declared private}}
  friend void foo2(D1::E1::F1); // expected-error{{'F1' is a private member of 'GH12361::D1::E1'}}
  // expected-note@#F1 {{implicitly declared private}}

  // FIXME: This should be diagnosed. We entered the function DC too early.
  friend void foo3() noexcept(sizeof(D1::E1::F1) == 1);
  friend void foo4() {
    D1::E1::F1 V;
  }
#if __cplusplus >= 202002L
  template <class T>
  friend void foo5(T)
    requires (sizeof(D1::E1::F1) == 1); // since-cxx20-error {{'F1' is a private member of 'GH12361::D1::E1'}}
  // since-cxx20-note@#F1 {{implicitly declared private}}
#endif
};

D1::E1::F1 foo1() { return D1::E1::F1(); }

}
