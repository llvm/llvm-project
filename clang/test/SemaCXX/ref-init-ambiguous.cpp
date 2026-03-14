// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

enum E2 { };

struct A { 
  operator E2&(); // expected-note 3 {{candidate function}}
};

struct B { 
  operator E2&(); // expected-note 3 {{candidate function}}
};

struct C : B, A { 
};

void test(C c) {
  const E2 &e2 = c; // expected-error {{reference initialization of type 'const E2 &' with initializer of type 'C' is ambiguous}}
}

void foo(const E2 &);// expected-note{{passing argument to parameter here}}

const E2 & re(C c) {
    foo(c); // expected-error {{reference initialization of type 'const E2 &' with initializer of type 'C' is ambiguous}}

    return c; // expected-error {{reference initialization of type 'const E2 &' with initializer of type 'C' is ambiguous}}
}

namespace CWG2352 {
  void f(const int * const &) = delete;
  void f(int *);

  void g(int * &);
  void g(const int *) = delete;

  void h1(int *const * const &);
  void h1(const int *const *) = delete;
  void h2(const int *const * const &) = delete;
  void h2(int *const *);

  void test() {
    int *x;
    // Under CWG2352, this became ambiguous. We order by qualification
    // conversion even when comparing a reference binding to a
    // non-reference-binding.
    f(x);
    g(x);
    h1(&x);
    h2(&x);
  }
}
