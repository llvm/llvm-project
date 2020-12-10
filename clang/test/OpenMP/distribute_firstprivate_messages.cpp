// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

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

extern int omp_default_mem_alloc;

S3 h;
#pragma omp threadprivate(h) // expected-note {{defined as threadprivate or thread local}}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  S6 p;
  int i, z;
  int &j = i;
  #pragma omp distribute firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp distribute firstprivate ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp distribute firstprivate () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate (argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate (a, b, c, d, f) // expected-error {{firstprivate variable with incomplete type 'S1'}} expected-warning {{Type 'const S2' is not trivially copyable and not guaranteed to be mapped correctly}} expected-warning {{Type 'const S3' is not trivially copyable and not guaranteed to be mapped correctly}} expected-error {{incomplete type 'S1' where a complete type is required}} expected-error {{no matching constructor for initialization of 'S3'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate (argv[1]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate(ba) // expected-warning {{Type 'const S2 [5]' is not trivially copyable and not guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate(ca) // expected-error {{no matching constructor for initialization of 'S3'}} expected-warning {{Type 'const S3 [5]' is not trivially copyable and not guaranteed to be mapped correctly}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate(da, z)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate(S2::S2s)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate(S2::S2sc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute private(i), firstprivate(i) // expected-error {{private variable cannot be firstprivate}} expected-note{{defined as private}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams shared(i)
  #pragma omp distribute firstprivate(i)
  for (j = 0; j < argc; ++j) foo();
  #pragma omp target
  #pragma omp teams shared(i)
  #pragma omp distribute firstprivate(i) // expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) foo(); // expected-error {{loop iteration variable in the associated loop of 'omp distribute' directive may not be firstprivate, predetermined as private}}
  #pragma omp target
  #pragma omp teams private(argc) // expected-note {{defined as private}}
  #pragma omp distribute firstprivate(argc) // expected-error {{firstprivate variable must be shared}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams reduction(+:argc) // expected-note {{defined as reduction}}
  #pragma omp distribute firstprivate(argc) // expected-error {{firstprivate variable must be shared}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute firstprivate(j)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute lastprivate(argc), firstprivate(argc) // expected-error {{lastprivate variable cannot be firstprivate}} expected-note{{defined as lastprivate}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target
  #pragma omp teams
#pragma omp distribute firstprivate(argc), lastprivate(argc)  // expected-error {{firstprivate variable cannot be lastprivate}} expected-note{{defined as firstprivate}}
  for (i = 0; i < argc; ++i) foo();
  return 0;
}
