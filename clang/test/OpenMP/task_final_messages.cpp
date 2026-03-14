// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T z;
  #pragma omp task final // expected-error {{expected '(' after 'final'}}
  #pragma omp task final ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final () // expected-error {{expected expression}}
  #pragma omp task final (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final (argc + z)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task final (argc > 0 ? argv[1] : argv[2])
  #pragma omp task final (foobool(argc)), final (true) // expected-error {{directive '#pragma omp task' cannot contain more than one 'final' clause}}
  #pragma omp task final (S) // expected-error {{'S' does not refer to a value}}
  #pragma omp task final (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final(argc)
  foo();

  return 0;
}

int main(int argc, char **argv) {
  int z;
  #pragma omp task final // expected-error {{expected '(' after 'final'}}
  #pragma omp task final ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final () // expected-error {{expected expression}}
  #pragma omp task final (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final (z+argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task final (argc > 0 ? argv[1] : argv[2])
  #pragma omp task final (foobool(argc)), final (true) // expected-error {{directive '#pragma omp task' cannot contain more than one 'final' clause}}
  #pragma omp task final (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp task final (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task final(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();

  return tmain(argc, argv);
}
