// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -ferror-limit 100 -o - %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

template <typename T, int C> // expected-note {{declared here}}
T tmain(T argc) {
  char **a;
#pragma omp target
#pragma omp teams distribute num_teams(C)
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(T) // expected-error {{'T' does not refer to a value}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams // expected-error {{expected '(' after 'num_teams'}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams() // expected-error {{expected expression}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute' are ignored}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(argc > 0 ? a[1] : a[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(argc + argc)
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(argc), num_teams (argc+1) // expected-error {{directive '#pragma omp teams distribute' cannot contain more than one 'num_teams' clause}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(S1) // expected-error {{'S1' does not refer to a value}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(-2) // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(-10u)
  for (int i=0; i<100; i++) foo();
#pragma omp target
#pragma omp teams distribute num_teams(3.14) // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'double'}}
  for (int i=0; i<100; i++) foo();

  return 0;
}

int main(int argc, char **argv) {
#pragma omp target
#pragma omp teams distribute num_teams // expected-error {{expected '(' after 'num_teams'}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams () // expected-error {{expected expression}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute' are ignored}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (argc + argc)
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (argc), num_teams (argc+1) // expected-error {{directive '#pragma omp teams distribute' cannot contain more than one 'num_teams' clause}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (S1) // expected-error {{'S1' does not refer to a value}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (-2) // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (-10u)
  for (int i=0; i<100; i++) foo();

#pragma omp target
#pragma omp teams distribute num_teams (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  for (int i=0; i<100; i++) foo();

  return tmain<int, 10>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 10>' requested here}}
}
