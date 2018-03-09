// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

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
  S2() : a(0) {}
  S2(const S2 &s2) : a(s2.a) {}
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
  S3 &operator=(const S3 &s3);
  
public:
  S3() : a(0) {} // expected-note 2 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  S3(S3 &s3) : a(s3.a) {} // expected-note 2 {{candidate constructor not viable: 1st argument ('const S3') would lose const qualifier}}
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
class S6 {
  int a;
public:
  S6() : a(0) { }
};

S3 h;
#pragma omp threadprivate(h) // expected-note {{defined as threadprivate or thread local}}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  S6 p;
  int i;
  int &j = i;

#pragma omp target teams distribute parallel for firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate (argc)
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate (a, b, c, d, f) // expected-error {{firstprivate variable with incomplete type 'S1'}} expected-error {{no matching constructor for initialization of 'S3'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate (argv[1]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate(ba)
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate(ca) // expected-error {{no matching constructor for initialization of 'S3'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate(da)
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate(S2::S2s)
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate(S2::S2sc)
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for private(i), firstprivate(i) // expected-error {{private variable cannot be firstprivate}} expected-note {{defined as private}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute parallel for firstprivate(i)
  for (j = 0; j < argc; ++j) foo();

#pragma omp target teams distribute parallel for firstprivate(i) // expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) foo(); // expected-error {{loop iteration variable in the associated loop of 'omp target teams distribute parallel for' directive may not be firstprivate, predetermined as private}}

#pragma omp target teams distribute parallel for firstprivate(j)
  for (i = 0; i < argc; ++i) foo();

// expected-error@+1 {{lastprivate variable cannot be firstprivate}} expected-note@+1 {{defined as lastprivate}}
#pragma omp target teams distribute parallel for lastprivate(argc), firstprivate(argc)
  for (i = 0; i < argc; ++i) foo();

  return 0;
}
