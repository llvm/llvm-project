// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2 & operator =(S2 &s2) { return *this; }
};
class S3 {
  int a;
public:
  S3():a(0) { }
  S3 &operator =(S3 &s3) { return *this; }
};
class S4 {
  int a;
  S4();
  S4 &operator =(const S4 &s4); // expected-note {{implicitly declared private here}}
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
  S5 &operator =(const S5 &s5) { return *this; } // expected-note {{implicitly declared private here}}
public:
  S5(int v):a(v) { }
};
template <class T>
class ST {
public:
  static T s;
};

namespace A {
double x;
#pragma omp threadprivate(x)
}
namespace B {
using A::x;
}

S2 k;
S3 h;
S4 l(3);
S5 m(4);
#pragma omp threadprivate(h, k, l, m)

int main(int argc, char **argv) {
  int i;
  #pragma omp parallel copyin // expected-error {{expected '(' after 'copyin'}}
  #pragma omp parallel copyin ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel copyin () // expected-error {{expected expression}}
  #pragma omp parallel copyin (k // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel copyin (h, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel copyin (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  #pragma omp parallel copyin (l) // expected-error {{'operator=' is a private member of 'S4'}}
  #pragma omp parallel copyin (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp parallel copyin (argv[1]) // expected-error {{expected variable name}}
  #pragma omp parallel copyin(i) // expected-error {{copyin variable must be threadprivate}}
  #pragma omp parallel copyin(m) // expected-error {{'operator=' is a private member of 'S5'}}
  #pragma omp parallel copyin(ST<int>::s) // expected-error {{copyin variable must be threadprivate}}
  #pragma omp parallel copyin(B::x)
  foo();

  return 0;
}
