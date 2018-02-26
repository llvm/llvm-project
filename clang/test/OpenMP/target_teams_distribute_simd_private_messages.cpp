// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}} expected-note {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  static float S2s; // expected-note {{predetermined as shared}}
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
};
const S3 c; // expected-note {{predetermined as shared}}
const S3 ca[5]; // expected-note {{predetermined as shared}}
extern const int f;  // expected-note {{predetermined as shared}}
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

S3 h;
#pragma omp threadprivate(h) // expected-note {{defined as threadprivate or thread local}}


int main(int argc, char **argv) {
  const int d = 5;  // expected-note {{predetermined as shared}}
  const int da[5] = { 0 }; // expected-note {{predetermined as shared}}
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;

#pragma omp target teams distribute simd private // expected-error {{expected '(' after 'private'}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private (argc)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private (a, b, c, d, f) // expected-error {{private variable with incomplete type 'S1'}} expected-error 3 {{shared variable cannot be private}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private(ba)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private(ca) // expected-error {{shared variable cannot be private}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private(da) // expected-error {{shared variable cannot be private}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private(S2::S2s) // expected-error {{shared variable cannot be private}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd shared(i)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd firstprivate(i), private(i) // expected-error {{firstprivate variable cannot be private}} expected-note {{defined as firstprivate}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd private(j)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target teams distribute simd reduction(+:i)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp distribute private(i)
  for (int k = 0; k < 10; ++k) {
    #pragma omp target teams distribute simd private(i)
    for (int x = 0; x < 10; ++x) foo();
  }

#pragma omp target teams distribute simd firstprivate(i)
  for (int k = 0; k < 10; ++k) {
  }

#pragma omp target teams distribute simd private(j) map(j) // expected-error {{private variable cannot be in a map clause in '#pragma omp target teams distribute simd' directive}} expected-note {{defined as private}}
  for (int k = 0; k < argc; ++k) ++k;

  return 0;
}
