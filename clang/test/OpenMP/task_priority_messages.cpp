// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  #pragma omp task priority // expected-error {{expected '(' after 'priority'}}
  #pragma omp task priority ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority () // expected-error {{expected expression}}
  #pragma omp task priority (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority (argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task priority (argc > 0 ? argv[1][0] : argv[2][argc])
  #pragma omp task priority (foobool(argc)), priority (true) // expected-error {{directive '#pragma omp task' cannot contain more than one 'priority' clause}}
  #pragma omp task priority (S) // expected-error {{'S' does not refer to a value}}
  #pragma omp task priority (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority(0)
  #pragma omp task priority(-1) // expected-error {{argument to 'priority' clause must be a non-negative integer value}}
  foo();

  return 0;
}

int main(int argc, char **argv) {
  #pragma omp task priority // expected-error {{expected '(' after 'priority'}}
  #pragma omp task priority ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority () // expected-error {{expected expression}}
  #pragma omp task priority (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority (argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task priority (argc > 0 ? argv[1][0] : argv[2][argc])
  #pragma omp task priority (foobool(argc)), priority (true) // expected-error {{directive '#pragma omp task' cannot contain more than one 'priority' clause}}
  #pragma omp task priority (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp task priority (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task priority(0)
  #pragma omp task priority(-1) // expected-error {{argument to 'priority' clause must be a non-negative integer value}}
  foo();

  return tmain(argc, argv);
}
