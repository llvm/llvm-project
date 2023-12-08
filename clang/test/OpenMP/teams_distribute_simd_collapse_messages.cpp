// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 %s -Wuninitialized

// expected-note@* 0+{{declared here}}

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1;

template <class T, typename S, int N, int ST>
T tmain(T argc, S **argv) {
#pragma omp target
#pragma omp teams distribute simd collapse // expected-error {{expected '(' after 'collapse'}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams distribute simd collapse ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)

    argv[0][i] = argv[0][i] - argv[0][i-ST];
#pragma omp target
#pragma omp teams distribute simd collapse () // expected-error {{expected expression}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
// expected-error@+2 2 {{integral constant expression}} expected-note@+2 0+{{constant expression}}
#pragma omp target
#pragma omp teams distribute simd collapse (argc 
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+2 2 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp target
#pragma omp teams distribute simd collapse (ST // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams distribute simd collapse (1)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute simd' are ignored}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams distribute simd collapse ((ST > 0) ? 1 + ST : 2) // expected-note 2 {{as specified in 'collapse' clause}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST]; // expected-error 2 {{expected 2 for loops after '#pragma omp teams distribute simd', but found only 1}}

// expected-error@+4 2 {{directive '#pragma omp teams distribute simd' cannot contain more than one 'collapse' clause}}
// expected-error@+3 {{argument to 'collapse' clause must be a strictly positive integer value}}
// expected-error@+2 2 {{integral constant expression}} expected-note@+2 0+{{constant expression}}
#pragma omp target
#pragma omp teams distribute simd collapse (foobool(argc)), collapse (true), collapse (-5)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp distribute collapse (S) // expected-error {{'S' does not refer to a value}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#if __cplusplus <= 199711L
  // expected-error@+5 2 {{integral constant expression}} expected-note@+5 0+{{constant expression}}
#else
  // expected-error@+3 2 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#endif
#pragma omp target
#pragma omp teams distribute simd collapse (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams distribute simd collapse (1)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams distribute simd collapse (N) // expected-error {{argument to 'collapse' clause must be a strictly positive integer value}}
  for (T i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams distribute simd collapse (2) // expected-note {{as specified in 'collapse' clause}}
  foo(); // expected-error {{expected 2 for loops after '#pragma omp teams distribute simd'}}
  return argc;
}

int main(int argc, char **argv) {
#pragma omp target
#pragma omp teams distribute simd collapse // expected-error {{expected '(' after 'collapse'}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams distribute simd collapse ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams distribute simd collapse () // expected-error {{expected expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams distribute simd collapse (4 // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-note {{as specified in 'collapse' clause}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4]; // expected-error {{expected 4 for loops after '#pragma omp teams distribute simd', but found only 1}}

#pragma omp target
#pragma omp teams distribute simd collapse (2+2)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute simd' are ignored}}  expected-note {{as specified in 'collapse' clause}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4]; // expected-error {{expected 4 for loops after '#pragma omp teams distribute simd', but found only 1}}

#pragma omp target
#pragma omp teams distribute simd collapse (foobool(1) > 0 ? 1 : 2) // expected-error {{integral constant expression}} expected-note 0+{{constant expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+4 {{integral constant expression}} expected-note@+4 0+{{constant expression}}
// expected-error@+3 2 {{directive '#pragma omp teams distribute simd' cannot contain more than one 'collapse' clause}}
// expected-error@+2 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp target
#pragma omp teams distribute simd collapse (foobool(argc)), collapse (true), collapse (-5) 
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams distribute simd collapse (S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#if __cplusplus >= 201103L
  // expected-error@+5 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#else
  // expected-error@+3 {{integral constant expression}} expected-note@+3 0+{{constant expression}}
#endif
#pragma omp target
#pragma omp teams distribute simd collapse (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+4 {{statement after '#pragma omp teams distribute simd' must be a for loop}}
// expected-note@+2 {{in instantiation of function template specialization 'tmain<int, char, -1, -2>' requested here}}
#pragma omp target
#pragma omp teams distribute simd collapse(collapse(tmain<int, char, -1, -2>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  foo();

#pragma omp target
#pragma omp teams distribute simd collapse (2) // expected-note {{as specified in 'collapse' clause}}
  foo(); // expected-error {{expected 2 for loops after '#pragma omp teams distribute simd'}}

// expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, 1, 0>' requested here}}
  return tmain<int, char, 1, 0>(argc, argv);
}

