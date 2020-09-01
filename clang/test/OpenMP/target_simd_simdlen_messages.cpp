// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 %s -Wuninitialized

void foo() {
}

#if __cplusplus >= 201103L
// expected-note@+2 4 {{declared here}}
#endif
bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, typename S, int N, int ST> // expected-note {{declared here}}
T tmain(T argc, S **argv) { //expected-note 2 {{declared here}}
#pragma omp target simd simdlen // expected-error {{expected '(' after 'simdlen'}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen () // expected-error {{expected expression}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
// expected-error@+2 2 {{integral constant expression}}
// expected-note@+1 2 {{read of non-const variable 'argc' is not allowed in a constant expression}}
#pragma omp target simd simdlen (argc
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp target simd simdlen (ST // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen (1)) // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen ((ST > 0) ? 1 + ST : 2)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

  // expected-error@+6 2 {{directive '#pragma omp target simd' cannot contain more than one 'simdlen' clause}}
  // expected-error@+5 {{argument to 'simdlen' clause must be a strictly positive integer value}}
  // expected-error@+4 2 {{integral constant expression}}
#if __cplusplus >= 201103L
  // expected-note@+2 2 {{non-constexpr function 'foobool' cannot be used in a constant expression}}
#endif
#pragma omp target simd simdlen (foobool(argc)), simdlen (true), simdlen (-5)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen (S) // expected-error {{'S' does not refer to a value}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#if __cplusplus <= 199711L
  // expected-error@+4 2 {{integral constant expression}}
#else
  // expected-error@+2 2 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#endif
#pragma omp target simd simdlen (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

 #pragma omp target simd simdlen (4)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

  #pragma omp target simd simdlen (N) // expected-error {{argument to 'simdlen' clause must be a strictly positive integer value}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen (2), safelen (4) // OK
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen (4), safelen (4) // OK
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target simd simdlen (8), safelen (4) // expected-error{{the value of 'simdlen' parameter must be less than or equal to the value of the 'safelen' parameter}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

  return argc;
}

int main(int argc, char **argv) {
#pragma omp target simd simdlen // expected-error {{expected '(' after 'simdlen'}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target simd simdlen ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target simd simdlen () // expected-error {{expected expression}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target simd simdlen (4 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target simd simdlen (2+2)) // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

  // expected-error@+4 {{integral constant expression}}
#if __cplusplus >= 201103L
  // expected-note@+2 {{non-constexpr function 'foobool' cannot be used in a constant expression}}
#endif
#pragma omp target simd simdlen (foobool(1) > 0 ? 1 : 2)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

  // expected-error@+6 {{integral constant expression}}
#if __cplusplus >= 201103L
  // expected-note@+4 {{non-constexpr function 'foobool' cannot be used in a constant expression}}
#endif
// expected-error@+2 2 {{directive '#pragma omp target simd' cannot contain more than one 'simdlen' clause}}
// expected-error@+1 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp target simd simdlen (foobool(argc)), simdlen (true), simdlen (-5) 
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target simd simdlen (S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#if __cplusplus <= 199711L
  // expected-error@+4 {{integral constant expression}}
#else
  // expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#endif
#pragma omp target simd simdlen (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+3 {{statement after '#pragma omp target simd' must be a for loop}}
// expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, -1, -2>' requested here}}
#pragma omp target simd simdlen(simdlen(tmain<int, char, -1, -2>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  foo();

#pragma omp target simd simdlen (2), safelen (4) // OK
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target simd simdlen (4), safelen (4) // OK
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target simd simdlen (8), safelen (4) // expected-error{{the value of 'simdlen' parameter must be less than or equal to the value of the 'safelen' parameter}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

  // expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, 12, 4>' requested here}}
  return tmain<int, char, 12, 4>(argc, argv);
}

