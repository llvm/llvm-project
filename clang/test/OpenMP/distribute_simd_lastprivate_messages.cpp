// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}} expected-note 2 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
  S2(S2 &s2) : a(s2.a) {}
  const S2 &operator =(const S2&) const;
  S2 &operator =(const S2&);
  static float S2s; // expected-note {{static data member is predetermined as shared}}
  static const float S2sc; // expected-note {{static data member is predetermined as shared}}
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
  S3 &operator=(const S3 &s3); // expected-note {{implicitly declared private here}}

public:
  S3() : a(0) {}
  S3(S3 &s3) : a(s3.a) {}
};
const S3 c;         // expected-note {{global variable is predetermined as shared}}
const S3 ca[5];     // expected-note {{global variable is predetermined as shared}}
extern const int f; // expected-note {{global variable is predetermined as shared}}
class S4 {
  int a;
  S4();             // expected-note 3 {{implicitly declared private here}}
  S4(const S4 &s4);

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5() : a(0) {} // expected-note {{implicitly declared private here}}

public:
  S5(const S5 &s5) : a(s5.a) {}
  S5(int v) : a(v) {}
};
class S6 {
  int a;
  S6() : a(0) {} // expected-note {{implicitly declared private here}}

public:
  S6(const S6 &s6) : a(s6.a) {}
  S6(int v) : a(v) {}
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template <class I, class C>
int foomain(int argc, char **argv) {
  I e(4);
  I g(5);
  int i;
  int &j = i;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate // expected-error {{expected '(' after 'lastprivate'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate() // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(a, b) // expected-error {{lastprivate variable with incomplete type 'S1'}} expected-warning {{Non-trivial type 'const S2' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(e, g) // expected-error 2 {{calling a private constructor of class 'S4'}} expected-warning 2 {{Non-trivial type 'S4' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(h) // expected-error {{threadprivate or thread local variable cannot be lastprivate}}
  for (int k = 0; k < argc; ++k)
    ++k;

  int v = 0;
#pragma omp target
#pragma omp teams
  {
#pragma omp distribute simd lastprivate(i)
    for (int k = 0; k < argc; ++k) {
      i = k;
      v += i;
    }
  }
#pragma omp target
#pragma omp teams private(i)
#pragma omp distribute simd lastprivate(j)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(i)
  for (int k = 0; k < argc; ++k)
    ++k;
  return 0;
}

void bar(S4 a[2]) {
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(a)
  for (int i = 0; i < 2; ++i)
    foo();
}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  const int d = 5;       // expected-note {{constant variable is predetermined as shared}}
  const int da[5] = {0}; // expected-note {{constant variable is predetermined as shared}}
  S4 e(4);
  S5 g(5);
  S3 m;
  S6 n(2);
  int i;
  int &j = i;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate // expected-error {{expected '(' after 'lastprivate'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate() // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argc)
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(a, b, c, d, f) // expected-error {{lastprivate variable with incomplete type 'S1'}} expected-error 3 {{shared variable cannot be lastprivate}} expected-error {{incomplete type 'S1' where a complete type is required}} expected-warning {{Non-trivial type 'const S2' is mapped, only trivial types are guaranteed to be mapped correctly}} expected-warning {{Non-trivial type 'const S3' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(argv[1]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(2 * 2) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(ba) // expected-warning {{Non-trivial type 'const S2 [5]' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(ca) // expected-error {{shared variable cannot be lastprivate}} expected-warning {{Non-trivial type 'const S3 [5]' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(da) // expected-error {{shared variable cannot be lastprivate}}
  for (i = 0; i < argc; ++i)
    foo();
  int xa;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(xa) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(S2::S2s) // expected-error {{shared variable cannot be lastprivate}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(S2::S2sc) // expected-error {{shared variable cannot be lastprivate}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd safelen(5) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}} expected-warning {{Non-trivial type 'S4' is mapped, only trivial types are guaranteed to be mapped correctly}} expected-warning {{Non-trivial type 'S5' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(m) // expected-error {{'operator=' is a private member of 'S3'}} expected-warning {{Non-trivial type 'S3' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(h) // expected-error {{threadprivate or thread local variable cannot be lastprivate}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(B::x) // expected-error {{threadprivate or thread local variable cannot be lastprivate}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd private(xa), lastprivate(xa) // expected-error {{private variable cannot be lastprivate}} expected-note {{defined as private}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(i) // expected-note {{defined as lastprivate}}
  for (i = 0; i < argc; ++i) // expected-error{{loop iteration variable in the associated loop of 'omp distribute simd' directive may not be lastprivate, predetermined as linear}}
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(xa)
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(xa)
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(j)
  for (i = 0; i < argc; ++i)
    foo();
// expected-error@+3 {{firstprivate variable cannot be lastprivate}} expected-note@+3 {{defined as firstprivate}}
#pragma omp target
#pragma omp teams
#pragma omp distribute simd firstprivate(m) lastprivate(m) // expected-warning {{Non-trivial type 'S3' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i)
    foo();
// expected-error@+3 {{lastprivate variable cannot be firstprivate}} expected-note@+3 {{defined as lastprivate}}
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(n) firstprivate(n) // expected-error {{calling a private constructor of class 'S6'}} expected-warning {{Non-trivial type 'S6' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i)
    foo();
  static int si;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(si) // OK
  for (i = 0; i < argc; ++i)
    si = i + 1;
  return foomain<S4, S5>(argc, argv); // expected-note {{in instantiation of function template specialization 'foomain<S4, S5>' requested here}}
}
