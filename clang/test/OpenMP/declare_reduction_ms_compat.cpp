// RUN: %clang_cc1 -verify -fopenmp -fms-compatibility %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fms-compatibility %s

struct foo {};
struct has_baz {
  typedef int baz;
};

template <typename T> struct S : T {
  virtual void bar() {
#pragma omp declare reduction(foo : baz : omp_out)
    // expected-warning@-1 {{use of undeclared identifier 'baz'; unqualified lookup into dependent bases of class template 'S' is a Microsoft extension}}
    // expected-error@-2 {{no type named 'baz' in 'S<foo>'}}
  }
};

template <typename T> struct NotInstantiated : T {
  virtual void bar() {
#pragma omp declare reduction(foo : baz : omp_out)
    // expected-warning@-1 {{use of undeclared identifier 'baz'; unqualified lookup into dependent bases of class template 'NotInstantiated' is a Microsoft extension}}
  }
};

S<foo> s; // expected-note {{in instantiation of member function 'S<foo>::bar' requested here}}
S<has_baz> t;
