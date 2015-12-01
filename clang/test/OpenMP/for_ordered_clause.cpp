// RUN: %clang_cc1 -verify -fopenmp %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, typename S, int N, int ST> // expected-note {{declared here}}
T tmain(T argc, S **argv) {                   //expected-note 2 {{declared here}}
#pragma omp for ordered
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered() // expected-error {{expected expression}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
// expected-error@+2 2 {{expression is not an integral constant expression}}
// expected-note@+1 2 {{read of non-const variable 'argc' is not allowed in a constant expression}}
#pragma omp for ordered(argc
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 2 {{argument to 'ordered' clause must be a strictly positive integer value}}
#pragma omp for ordered(ST // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered(1)) // expected-warning {{extra tokens at the end of '#pragma omp for' are ignored}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered((ST > 0) ? 1 + ST : 2) // expected-note 2 {{as specified in 'ordered' clause}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST]; // expected-error 2 {{expected 2 for loops after '#pragma omp for', but found only 1}}
// expected-error@+3 2 {{directive '#pragma omp for' cannot contain more than one 'ordered' clause}}
// expected-error@+2 2 {{argument to 'ordered' clause must be a strictly positive integer value}}
// expected-error@+1 2 {{expression is not an integral constant expression}}
#pragma omp for ordered(foobool(argc)), ordered(true), ordered(-5)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered(S) // expected-error {{'S' does not refer to a value}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
// expected-error@+1 2 {{expression is not an integral constant expression}}
#pragma omp for ordered(argv[1] = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered(1)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered(N-1) // expected-error 2 {{argument to 'ordered' clause must be a strictly positive integer value}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered(N) // expected-error {{argument to 'ordered' clause must be a strictly positive integer value}}
  for (T i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i - ST];
#pragma omp for ordered(2) // expected-note {{as specified in 'ordered' clause}}
  foo();                    // expected-error {{expected 2 for loops after '#pragma omp for'}}
#pragma omp for ordered(N) collapse(N + 2) // expected-error {{the parameter of the 'ordered' clause must be greater than or equal to the parameter of the 'collapse' clause}} expected-note {{parameter of the 'collapse' clause}} expected-error {{argument to 'ordered' clause must be a strictly positive integer value}}
  for (int i = ST; i < N; i++)
    for (int j = ST; j < N; j++)
      for (int k = ST; k < N; k++)
        foo();
  return argc;
}

int main(int argc, char **argv) {
#pragma omp for ordered
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp for ordered( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp for ordered() // expected-error {{expected expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp for ordered(4 // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-note {{as specified in 'ordered' clause}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4]; // expected-error {{expected 4 for loops after '#pragma omp for', but found only 1}}
#pragma omp for ordered(2 + 2))              // expected-warning {{extra tokens at the end of '#pragma omp for' are ignored}}  expected-note {{as specified in 'ordered' clause}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];    // expected-error {{expected 4 for loops after '#pragma omp for', but found only 1}}
#pragma omp for ordered(foobool(1) > 0 ? 1 : 2) // expected-error {{expression is not an integral constant expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
// expected-error@+3 {{expression is not an integral constant expression}}
// expected-error@+2 2 {{directive '#pragma omp for' cannot contain more than one 'ordered' clause}}
// expected-error@+1 2 {{argument to 'ordered' clause must be a strictly positive integer value}}
#pragma omp for ordered(foobool(argc)), ordered(true), ordered(-5)
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp for ordered(S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
// expected-error@+1 {{expression is not an integral constant expression}}
#pragma omp for ordered(argv[1] = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
// expected-error@+3 {{statement after '#pragma omp for' must be a for loop}}
// expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, -1, -2>' requested here}}
#pragma omp for ordered(ordered(tmain < int, char, -1, -2 > (argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  foo();
#pragma omp for ordered(2) // expected-note {{as specified in 'ordered' clause}}
  foo();                    // expected-error {{expected 2 for loops after '#pragma omp for'}}
#pragma omp for ordered(0)              // expected-error {{argument to 'ordered' clause must be a strictly positive integer value}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i - 4];
#pragma omp for ordered(2) collapse(3) // expected-error {{the parameter of the 'ordered' clause must be greater than or equal to the parameter of the 'collapse' clause}} expected-note {{parameter of the 'collapse' clause}}
  for (int i = 0; i < 10; i++)
    for (int j = 0; j < 11; j++)
      for (int k = 0; k < 12; k++)
        foo();
  // expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, 1, 0>' requested here}}
  return tmain<int, char, 1, 0>(argc, argv);
}

