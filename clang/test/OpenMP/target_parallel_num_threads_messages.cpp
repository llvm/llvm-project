// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

#define redef_num_threads(a, b) num_threads(a)

template <class T, typename S, int N> // expected-note {{declared here}}
T tmain(T argc, S **argv) {
  #pragma omp target parallel num_threads // expected-error {{expected '(' after 'num_threads'}}
  foo();
  #pragma omp target parallel num_threads ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads () // expected-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel num_threads ((argc > 0) ? argv[1] : argv[2]) // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads (foobool(argc)), num_threads (true), num_threads (-5) // expected-error 2 {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}} expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads (S) // expected-error {{'S' does not refer to a value}}
  foo();
  #pragma omp target parallel num_threads (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads (argc)
  foo();
  #pragma omp target parallel num_threads (N) // expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel redef_num_threads (argc, argc)
  foo();

  return argc;
}

int main(int argc, char **argv) {
  #pragma omp target parallel num_threads // expected-error {{expected '(' after 'num_threads'}}
  foo();
  #pragma omp target parallel num_threads ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads () // expected-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel num_threads (argc > 0 ? argv[1] : argv[2]) // expected-error {{integral }}
  foo();
  #pragma omp target parallel num_threads (foobool(argc)), num_threads (true), num_threads (-5) // expected-error 2 {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}} expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target parallel num_threads (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads (num_threads(tmain<int, char, -1>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}} expected-note {{in instantiation of function template specialization 'tmain<int, char, -1>' requested here}}
  foo();
  #pragma omp target parallel redef_num_threads (argc, argc)
  foo();

  return tmain<int, char, 3>(argc, argv); // expected-note {{in instantiation of function template specialization 'tmain<int, char, 3>' requested here}}
}
