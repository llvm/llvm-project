// RUN: %clang_cc1 -verify -fopenmp %s
// RUN: %clang_cc1 -verify -fopenmp -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp -std=c++11 %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 %s

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

#pragma omp target teams distribute parallel for simd safelen // expected-error {{expected '(' after 'safelen'}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen () // expected-error {{expected expression}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen (argc // expected-note {{to match this '('}} expected-error 2 {{expression is not an integral constant expression}} expected-note 2 {{read of non-const variable 'argc' is not allowed in a constant expression}} expected-error {{expected ')'}}
  for (int i = ST; i < N; i++) 
    argv[0][i] = argv[0][i] - argv[0][i-ST];
  
#pragma omp target teams distribute parallel for simd safelen (ST // expected-error {{argument to 'safelen' clause must be a strictly positive integer value}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen (1)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for simd' are ignored}}
  for (int i = ST; i < N; i++)
     argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen ((ST > 0) ? 1 + ST : 2)
  for (int i = ST; i < N; i++) 
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#if __cplusplus >= 201103L
  // expected-note@+5 2 {{non-constexpr function 'foobool' cannot be used in a constant expression}}
#endif
// expected-error@+3 2 {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'safelen' clause}}
// expected-error@+2 {{argument to 'safelen' clause must be a strictly positive integer value}}
// expected-error@+1 2 {{expression is not an integral constant expression}}
#pragma omp target teams distribute parallel for simd safelen (foobool(argc)), safelen (true), safelen (-5)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen (S) // expected-error {{'S' does not refer to a value}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#if __cplusplus <= 199711L
  // expected-error@+4 2 {{expression is not an integral constant expression}}
#else
  // expected-error@+2 2 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#endif
#pragma omp target teams distribute parallel for simd safelen (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen (4)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd safelen (N) // expected-error {{argument to 'safelen' clause must be a strictly positive integer value}}
  for (T i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

  return argc;
}

int main(int argc, char **argv) {
#pragma omp target teams distribute parallel for simd safelen // expected-error {{expected '(' after 'safelen'}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target teams distribute parallel for simd safelen ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target teams distribute parallel for simd safelen () // expected-error {{expected expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target teams distribute parallel for simd safelen (4 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target teams distribute parallel for simd safelen (2+2)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for simd' are ignored}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];
  
#if __cplusplus >= 201103L
  // expected-note@+2 {{non-constexpr function 'foobool' cannot be used in a constant expression}}
#endif
#pragma omp target teams distribute parallel for simd safelen (foobool(1) > 0 ? 1 : 2) // expected-error {{expression is not an integral constant expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#if __cplusplus >= 201103L
  // expected-note@+5 {{non-constexpr function 'foobool' cannot be used in a constant expression}}
#endif
// expected-error@+3 {{argument to 'safelen' clause must be a strictly positive integer value}}
// expected-error@+2 2 {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'safelen' clause}}
// expected-error@+1 {{expression is not an integral constant expression}}
#pragma omp target teams distribute parallel for simd safelen (foobool(argc)), safelen (true), safelen (-5)
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target teams distribute parallel for simd safelen (S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#if __cplusplus <= 199711L
  // expected-error@+4 {{expression is not an integral constant expression}}
#else
  // expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#endif
#pragma omp target teams distribute parallel for simd safelen (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, -1, -2>' requested here}}
#pragma omp target teams distribute parallel for simd safelen(safelen(tmain<int, char, -1, -2>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  foo(); // expected-error {{statement after '#pragma omp target teams distribute parallel for simd' must be a for loop}}

  // expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, 12, 4>' requested here}}
  return tmain<int, char, 12, 4>(argc, argv);
}

