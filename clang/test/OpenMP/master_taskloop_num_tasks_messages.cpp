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
  #pragma omp master taskloop num_tasks // expected-error {{expected '(' after 'num_tasks'}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks () // expected-error {{expected expression}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc)) // expected-warning {{extra tokens at the end of '#pragma omp master taskloop' are ignored}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc > 0 ? argv[1][0] : argv[2][argc] + z)
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (foobool(argc)), num_tasks (true) // expected-error {{directive '#pragma omp master taskloop' cannot contain more than one 'num_tasks' clause}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (S) // expected-error {{'S' does not refer to a value}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks(0) // expected-error {{argument to 'num_tasks' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks(-1) // expected-error {{argument to 'num_tasks' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks(argc) grainsize(argc) // expected-error {{'grainsize' and 'num_tasks' clause are mutually exclusive and may not appear on the same directive}} expected-note {{'num_tasks' clause is specified here}}
  for (int i = 0; i < 10; ++i)
    foo();

  return 0;
}

int main(int argc, char **argv) {
  int z;
  #pragma omp master taskloop num_tasks // expected-error {{expected '(' after 'num_tasks'}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks () // expected-error {{expected expression}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc)) // expected-warning {{extra tokens at the end of '#pragma omp master taskloop' are ignored}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc > 0 ? argv[1][0] : argv[2][argc] - z)
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (foobool(argc)), num_tasks (true) // expected-error {{directive '#pragma omp master taskloop' cannot contain more than one 'num_tasks' clause}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks(0)  // expected-error {{argument to 'num_tasks' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks(-1) // expected-error {{argument to 'num_tasks' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop num_tasks(argc) grainsize(argc) // expected-error {{'grainsize' and 'num_tasks' clause are mutually exclusive and may not appear on the same directive}} expected-note {{'num_tasks' clause is specified here}}
  for (int i = 0; i < 10; ++i)
    foo();

  return tmain(argc, argv);
}
