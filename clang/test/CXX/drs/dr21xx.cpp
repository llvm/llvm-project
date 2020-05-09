// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-error@+1 {{variadic macro}}
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
#endif

namespace dr2103 { // dr2103: yes
  void f() {
    int a;
    int &r = a; // expected-note {{here}}
    struct Inner {
      void f() {
        int &s = r; // expected-error {{enclosing function}}
        (void)s;
      }
    };
  }
}

namespace dr2120 { // dr2120: 7
  struct A {};
  struct B : A {};
  struct C { A a; };
  struct D { C c[5]; };
  struct E : B { D d; };
  static_assert(__is_standard_layout(B), "");
  static_assert(__is_standard_layout(D), "");
  static_assert(!__is_standard_layout(E), "");
}

namespace dr2140 { // dr2140: 9
#if __cplusplus >= 201103L
  union U { int a; decltype(nullptr) b; };
  constexpr int *test(U u) {
    return u.b;
  }
  static_assert(!test({123}), "u.b should be valid even when b is inactive");
#endif
}

namespace dr2157 { // dr2157: 11
#if __cplusplus >= 201103L
  enum E : int;
  struct X {
    enum dr2157::E : int(); // expected-error {{only allows ':' in member enumeration declaration to introduce a fixed underlying type}}
  };
#endif
}

namespace dr2170 { // dr2170: 9
#if __cplusplus >= 201103L
  void f() {
    constexpr int arr[3] = {1, 2, 3}; // expected-note {{here}}
    struct S {
      int get(int n) { return arr[n]; }
      const int &get_ref(int n) { return arr[n]; } // expected-error {{enclosing function}}
      // FIXME: expected-warning@-1 {{reference to stack}}
    };
  }
#endif
}

namespace dr2180 { // dr2180: yes
  class A {
    A &operator=(const A &); // expected-note 0-2{{here}}
    A &operator=(A &&); // expected-note 0-2{{here}} expected-error 0-1{{extension}}
  };

  struct B : virtual A {
    B &operator=(const B &);
    B &operator=(B &&); // expected-error 0-1{{extension}}
    virtual void foo() = 0;
  };
#if __cplusplus < 201103L
  B &B::operator=(const B&) = default; // expected-error {{private member}} expected-error {{extension}} expected-note {{here}}
  B &B::operator=(B&&) = default; // expected-error {{private member}} expected-error 2{{extension}} expected-note {{here}}
#else
  B &B::operator=(const B&) = default; // expected-error {{would delete}} expected-note@-9{{inaccessible copy assignment}}
  B &B::operator=(B&&) = default; // expected-error {{would delete}} expected-note@-10{{inaccessible move assignment}}
#endif
}
