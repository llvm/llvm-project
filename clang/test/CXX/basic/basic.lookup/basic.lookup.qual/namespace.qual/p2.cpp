// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace Ints {
  int zero = 0; // expected-note {{candidate found by name lookup is 'Ints::zero'}}
  void f(int); // expected-note 3 {{candidate function}}
  void g(int);
}

namespace Floats {
  float zero = 0.0f; // expected-note {{candidate found by name lookup is 'Floats::zero'}}
  void f(float); // expected-note 3 {{candidate function}}
  void g(float);
}

namespace Numbers {
  using namespace Ints;
  using namespace Floats;
}

void test() {
  int i = Ints::zero;
  Ints::f(i);
  
  float f = Floats::zero;
  Floats::f(f);
  
  double n = Numbers::zero; // expected-error {{reference to 'zero' is ambiguous}}
  Numbers::f(n); // expected-error{{call to 'f' is ambiguous}}
  Numbers::f(i);
  Numbers::f(f);
}

namespace Numbers {
  struct Number { // expected-note 2 {{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
  // expected-note@-2 2 {{candidate constructor (the implicit move constructor) not viable}}
#endif

    explicit Number(double d) : d(d) {} // expected-note 2{{explicit constructor is not a candidate}}
    double d;
  };
  Number zero(0.0f);
  void g(Number); // expected-note 2{{passing argument to parameter here}}
}

void test2() {
  Numbers::Number n = Numbers::zero;
  Numbers::f(n); // expected-error {{no matching function for call to 'f'}}
  Numbers::g(n);
}

namespace Numbers2 {
  using Numbers::f;
  using Numbers::g;
}

void test3() {
  Numbers::Number n = Numbers::zero;
  Numbers2::f(n); // expected-error {{no matching function for call to 'f'}}
  Numbers2::g(n);

  int i = Ints::zero;
  Numbers2::f(i);
  Numbers2::g(i); // expected-error {{no viable conversion from 'int' to 'Number'}}

  float f = Floats::zero;
  Numbers2::f(f);
  Numbers2::g(f); // expected-error {{no viable conversion from 'float' to 'Number'}}
}

namespace inline_ns {
  int x; // expected-note 2{{found}}
  inline namespace A {
#if __cplusplus <= 199711L // C++03 or earlier
  // expected-warning@-2 {{inline namespaces are a C++11 feature}}
#endif

    int x; // expected-note 2{{found}}
    int y; // expected-note 2{{found}}
  }
  int y; // expected-note 2{{found}}
  int k1 = x + y; // expected-error 2{{ambiguous}}
  int k2 = inline_ns::x + inline_ns::y; // expected-error 2{{ambiguous}}
}
