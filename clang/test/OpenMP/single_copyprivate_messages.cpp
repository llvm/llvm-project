// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

void foo() {
}

struct S1; // expected-note 2 {{declared here}}
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
  S2 &operator=(S2 &s2) { return *this; }
};
class S3 {
  int a;

public:
  S3() : a(0) {}
  S3 &operator=(S3 &s3) { return *this; }
};
class S4 {
  int a;
  S4();
  S4 &operator=(const S4 &s4); // expected-note 3 {{implicitly declared private here}}

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5() : a(0) {}
  S5 &operator=(const S5 &s5) { return *this; } // expected-note 3 {{implicitly declared private here}}

public:
  S5(int v) : a(v) {}
};

S2 k;
S3 h;
S4 l(3);
S5 m(4);
#pragma omp threadprivate(h, k, l, m)

template <class T, class C>
T tmain(T argc, C **argv) {
  T i;
  static T TA;
#pragma omp parallel
#pragma omp single copyprivate // expected-error {{expected '(' after 'copyprivate'}}
#pragma omp parallel
#pragma omp single copyprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel
#pragma omp single copyprivate() // expected-error {{expected expression}}
#pragma omp parallel
#pragma omp single copyprivate(k // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel
#pragma omp single copyprivate(h, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel
#pragma omp single copyprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
#pragma omp parallel
#pragma omp single copyprivate(l) // expected-error 2 {{'operator=' is a private member of 'S4'}}
#pragma omp parallel
#pragma omp single copyprivate(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp parallel
#pragma omp single copyprivate(argv[1]) // expected-error {{expected variable name}}
#pragma omp parallel // expected-note {{implicitly determined as shared}}
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
#pragma omp parallel
#pragma omp single copyprivate(m) // expected-error 2 {{'operator=' is a private member of 'S5'}}
  foo();
#pragma omp parallel private(i)
  {
#pragma omp single copyprivate(i)
    foo();
  }
#pragma omp parallel shared(i) // expected-note {{defined as shared}}
  {
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
    foo();
  }
#pragma omp parallel private(i)
#pragma omp parallel default(shared) // expected-note {{implicitly determined as shared}}
  {
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
    foo();
  }
#pragma omp parallel private(i)
#pragma omp parallel // expected-note {{implicitly determined as shared}}
  {
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
    foo();
  }
#pragma omp parallel
#pragma omp single private(i) copyprivate(i) // expected-error {{private variable cannot be copyprivate}} expected-note {{defined as private}}
  foo();
#pragma omp parallel
#pragma omp single firstprivate(i) copyprivate(i) // expected-error {{firstprivate variable cannot be copyprivate}} expected-note {{defined as firstprivate}}
  foo();
#pragma omp parallel private(TA)
  {
#pragma omp single copyprivate(TA)
    TA = 99;
  }

  return T();
}

void bar(S4 a[2], int n, int b[n]) {
#pragma omp single copyprivate(a, b)
    foo();
}

namespace A {
double x;
#pragma omp threadprivate(x)
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  int i;
  static int intA;
#pragma omp parallel
#pragma omp single copyprivate // expected-error {{expected '(' after 'copyprivate'}}
#pragma omp parallel
#pragma omp single copyprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel
#pragma omp single copyprivate() // expected-error {{expected expression}}
#pragma omp parallel
#pragma omp single copyprivate(k // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel
#pragma omp single copyprivate(h, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel
#pragma omp single copyprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
#pragma omp parallel
#pragma omp single copyprivate(l, B::x) // expected-error {{'operator=' is a private member of 'S4'}}
#pragma omp parallel
#pragma omp single copyprivate(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp parallel
#pragma omp single copyprivate(argv[1]) // expected-error {{expected variable name}}
#pragma omp parallel // expected-note {{implicitly determined as shared}}
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
#pragma omp parallel
#pragma omp single copyprivate(m) // expected-error {{'operator=' is a private member of 'S5'}}
  foo();
#pragma omp parallel private(i)
  {
#pragma omp single copyprivate(i)
    foo();
  }
#pragma omp parallel shared(i) // expected-note {{defined as shared}}
  {
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
    foo();
  }
#pragma omp parallel private(i)
#pragma omp parallel default(shared) // expected-note {{implicitly determined as shared}}
  {
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
    foo();
  }
#pragma omp parallel private(i)
#pragma omp parallel // expected-note {{implicitly determined as shared}}
  {
#pragma omp single copyprivate(i) // expected-error {{copyprivate variable must be threadprivate or private in the enclosing context}}
    foo();
  }
#pragma omp parallel
#pragma omp single private(i) copyprivate(i) // expected-error {{private variable cannot be copyprivate}} expected-note {{defined as private}}
  foo();
#pragma omp parallel
#pragma omp single firstprivate(i) copyprivate(i) // expected-error {{firstprivate variable cannot be copyprivate}} expected-note {{defined as firstprivate}}
  foo();
#pragma omp single copyprivate(i) nowait // expected-error {{the 'copyprivate' clause must not be used with the 'nowait' clause}} expected-note {{'nowait' clause is here}}
  foo();
#pragma omp parallel private(intA)
  {
#pragma omp single copyprivate(intA)
    intA = 99;
  }

  return tmain(argc, argv); // expected-note {{in instantiation of function template specialization 'tmain<int, char>' requested here}}
}

extern void abort(void);

void
single(int a, int b) {
#pragma omp single copyprivate(a) copyprivate(b)
  {
    a = b = 5;
  }

  if (a != b)
    abort();
}

int parallel() {
#pragma omp parallel
  single(1, 2);

  return 0;
}
