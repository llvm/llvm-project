// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}} expected-note{{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(const S2 &s2):a(s2.a) { }
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
  S3(const S3 &s3):a(s3.a) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 {
  int a;
  S4();
  S4(const S4 &s4); // expected-note {{implicitly declared private here}}
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
  S5(const S5 &s5):a(s5.a) { } // expected-note {{implicitly declared private here}}
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
  static int m;
  #pragma omp target parallel firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  #pragma omp target parallel firstprivate ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target parallel firstprivate () // expected-error {{expected expression}}
  #pragma omp target parallel firstprivate (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target parallel firstprivate (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target parallel firstprivate (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  #pragma omp target parallel firstprivate (argc)
  #pragma omp target parallel firstprivate (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp target parallel firstprivate (a, b, c, d, f) // expected-error {{firstprivate variable with incomplete type 'S1'}}
  #pragma omp target parallel firstprivate (argv[1]) // expected-error {{expected variable name}}
  #pragma omp target parallel firstprivate(ba)
  #pragma omp target parallel firstprivate(ca)
  #pragma omp target parallel firstprivate(da)
  #pragma omp target parallel firstprivate(S2::S2s)
  #pragma omp target parallel firstprivate(S2::S2sc)
  #pragma omp target parallel firstprivate(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  #pragma omp target parallel firstprivate(h, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be firstprivate}}
  #pragma omp target parallel private(i), firstprivate(i) // expected-error {{private variable cannot be firstprivate}} expected-note{{defined as private}}
  foo();
  #pragma omp target parallel shared(i)
  #pragma omp target parallel firstprivate(i)
  #pragma omp target parallel firstprivate(j)
  #pragma omp target parallel firstprivate(m)
  foo();

  return 0;
}
