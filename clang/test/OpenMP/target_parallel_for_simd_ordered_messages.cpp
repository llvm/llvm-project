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
  int j;
#pragma omp target parallel for simd ordered
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp target parallel for simd ordered( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp target parallel for simd ordered() // expected-error {{expected expression}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+2 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target parallel for simd ordered(argc
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(ST // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(1)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel for simd' are ignored}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered((ST > 0) ? 1 + ST : 2)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+3 {{argument to 'ordered' clause must be a strictly positive integer value}}
// expected-error@+2 2 {{directive '#pragma omp target parallel for simd' cannot contain more than one 'ordered' clause}}
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(foobool(argc)), ordered(true), ordered(-5)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp target parallel for simd ordered(S) // expected-error {{'S' does not refer to a value}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];

// expected-error@+1 {{integral constant expression}} expected-note@+1 0+{{constant expression}}
#pragma omp target parallel for simd ordered(j = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(1)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(N)
  for (T i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(2)
  foo();
  return argc;
}

int main(int argc, char **argv) {
  int j;
#pragma omp target parallel for simd ordered
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp target parallel for simd ordered( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp target parallel for simd ordered() // expected-error {{expected expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(4 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(2 + 2))      // expected-warning {{extra tokens at the end of '#pragma omp target parallel for simd' are ignored}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];

#pragma omp target parallel for simd ordered(foobool(1) > 0 ? 1 : 2) // expected-error {{integral constant expression}} expected-note 0+{{constant expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];

// expected-error@+3 {{integral constant expression}} expected-note@+3 0+{{constant expression}}
// expected-error@+2 2 {{directive '#pragma omp target parallel for simd' cannot contain more than one 'ordered' clause}}
// expected-error@+1 {{argument to 'ordered' clause must be a strictly positive integer value}}
#pragma omp target parallel for simd ordered(foobool(argc)), ordered(true), ordered(-5)
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp target parallel for simd ordered(S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];

// expected-error@+1 {{integral constant expression}} expected-note@+1 0+{{constant expression}}
#pragma omp target parallel for simd ordered(j = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
// expected-error@+2 {{statement after '#pragma omp target parallel for simd' must be a for loop}}
#pragma omp target parallel for simd ordered(ordered(tmain < int, char, -1, -2 > (argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  foo();
// expected-error@+1 {{'ordered' clause with a parameter can not be specified in '#pragma omp target parallel for simd' directive}}
#pragma omp target parallel for simd ordered(2)
  foo();
  return tmain<int, char, 1, 0>(argc, argv);
}

