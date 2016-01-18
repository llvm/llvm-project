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
  static float S2s;
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
};
const S3 c; // expected-note {{global variable is predetermined as shared}}
const S3 ca[5]; // expected-note {{global variable is predetermined as shared}}
extern const int f; // expected-note {{global variable is predetermined as shared}}
class S4 {
  int a;
  S4(); // expected-note {{implicitly declared private here}}
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {} // expected-note {{implicitly declared private here}}
public:
  S5(int v):a(v) { }
};

int threadvar;
#pragma omp threadprivate(threadvar) // expected-note {{defined as threadprivate or thread local}}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  const int d = 5; // expected-note {{constant variable is predetermined as shared}}
  const int da[5] = { 0 }; // expected-note {{constant variable is predetermined as shared}}
  S4 e(4);
  S5 g[] = {5, 6};
  int i;
  int &j = i;
  #pragma omp parallel private // expected-error {{expected '(' after 'private'}}
  #pragma omp parallel private ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel private () // expected-error {{expected expression}}
  #pragma omp parallel private (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel private (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel private (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  #pragma omp parallel private (argc argv) // expected-error {{expected ',' or ')' in 'private' clause}}
  #pragma omp parallel private (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp parallel private (a, b, c, d, f) // expected-error {{a private variable with incomplete type 'S1'}} expected-error 3 {{shared variable cannot be private}}
  #pragma omp parallel private (argv[1]) // expected-error {{expected variable name}}
  #pragma omp parallel private(ba)
  #pragma omp parallel private(ca) // expected-error {{shared variable cannot be private}}
  #pragma omp parallel private(da) // expected-error {{shared variable cannot be private}}
  #pragma omp parallel private(S2::S2s)
  #pragma omp parallel private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  #pragma omp parallel private(threadvar, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be private}}
  #pragma omp parallel shared(i), private(i) // expected-error {{shared variable cannot be private}} expected-note {{defined as shared}}
  foo();
  #pragma omp parallel firstprivate(i) private(i) // expected-error {{firstprivate variable cannot be private}} expected-note {{defined as firstprivate}}
  foo();
  #pragma omp parallel private(i)
  #pragma omp parallel private(j)
  foo();
  #pragma omp parallel firstprivate(i)
  for (int k = 0; k < 10; ++k) {
    #pragma omp parallel private(i)
    foo();
  }
  static int m;
  #pragma omp parallel private(m) // OK
  foo();

  return 0;
}
