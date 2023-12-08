// RUN: %clang_cc1 -verify -fopenmp -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

template <typename T, int C> // expected-note {{declared here}}
T tmain(T argc) {
  char **a;
  T z;
#pragma omp target
#pragma omp teams distribute simd thread_limit(C)
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(T) // expected-error {{'T' does not refer to a value}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit // expected-error {{expected '(' after 'thread_limit'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit() // expected-error {{expected expression}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute simd' are ignored}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(argc > 0 ? a[1] : a[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(argc + argc + z)
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(argc), thread_limit (argc+1) // expected-error {{directive '#pragma omp teams distribute simd' cannot contain more than one 'thread_limit' clause}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(S1) // expected-error {{'S1' does not refer to a value}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(-2) // expected-error {{argument to 'thread_limit' clause must be a strictly positive integer value}}
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(-10u)
  for (int j=0; j<100; j++) foo();
#pragma omp target
#pragma omp teams distribute simd thread_limit(3.14) // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'double'}}
  for (int j=0; j<100; j++) foo();

  return 0;
}

int main(int argc, char **argv) {
  int z;
#pragma omp target
#pragma omp teams distribute simd thread_limit // expected-error {{expected '(' after 'thread_limit'}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit () // expected-error {{expected expression}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute simd' are ignored}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (argc + argc + z)
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (argc), thread_limit (argc+1) // expected-error {{directive '#pragma omp teams distribute simd' cannot contain more than one 'thread_limit' clause}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (S1) // expected-error {{'S1' does not refer to a value}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (-2) // expected-error {{argument to 'thread_limit' clause must be a strictly positive integer value}}
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (-10u)
  for (int j=0; j<100; j++) foo();

#pragma omp target
#pragma omp teams distribute simd thread_limit (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  for (int j=0; j<100; j++) foo();

  return tmain<int, 10>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 10>' requested here}}
}
