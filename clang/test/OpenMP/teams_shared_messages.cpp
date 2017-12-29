// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
  S3(S3 &s3):a(s3.a) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 {
  int a;
  S4();
  S4(const S4 &s4);
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
  S5(const S5 &s5):a(s5.a) { }
public:
  S5(int v):a(v) { }
};

S3 h;
#pragma omp threadprivate(h) // expected-note {{defined as threadprivate or thread local}}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;
  #pragma omp target
  #pragma omp teams shared // expected-error {{expected '(' after 'shared'}}
  foo();
  #pragma omp target
  #pragma omp teams shared ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target
  #pragma omp teams shared () // expected-error {{expected expression}}
  foo();
  #pragma omp target
  #pragma omp teams shared (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target
  #pragma omp teams shared (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target
  #pragma omp teams shared (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  foo();
  #pragma omp target
  #pragma omp teams shared (argc)
  foo();
  #pragma omp target
  #pragma omp teams shared (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target
  #pragma omp teams shared (a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}}
  foo();
  #pragma omp target
  #pragma omp teams shared (argv[1]) // expected-error {{expected variable name}}
  foo();
  #pragma omp target
  #pragma omp teams shared(ba)
  foo();
  #pragma omp target
  #pragma omp teams shared(ca)
  foo();
  #pragma omp target
  #pragma omp teams shared(da)
  foo();
  #pragma omp target
  #pragma omp teams shared(e, g)
  foo();
  #pragma omp target
  #pragma omp teams shared(h, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be shared}}
  foo();
  #pragma omp target
  #pragma omp teams private(i), shared(i) // expected-error {{private variable cannot be shared}} expected-note {{defined as private}}
  foo();
  #pragma omp target
  #pragma omp teams firstprivate(i), shared(i) // expected-error {{firstprivate variable cannot be shared}} expected-note {{defined as firstprivate}}
  foo();
  #pragma omp target
  #pragma omp teams private(i)
  foo();
  #pragma omp target
  #pragma omp teams shared(i)
  foo();
  #pragma omp target
  #pragma omp teams shared(j)
  foo();
  #pragma omp target
  #pragma omp teams firstprivate(i)
  foo();
  #pragma omp target
  #pragma omp teams shared(i)
  foo();
  #pragma omp target
  #pragma omp teams shared(j)
  foo();

  return 0;
}
