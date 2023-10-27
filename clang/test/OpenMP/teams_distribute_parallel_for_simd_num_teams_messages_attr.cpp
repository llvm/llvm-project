// RUN: %clang_cc1 -verify -fsyntax-only -fopenmp -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fsyntax-only -fopenmp-simd -std=c++11 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

template <typename T, int C> // expected-note {{declared here}}
T tmain(T argc) {
  char **a;
  T k;
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(C)))]]
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(T)))]] // expected-error {{'T' does not refer to a value}}
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams))]] // expected-error {{expected '(' after 'num_teams'}}
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams()))]] // expected-error {{expected expression}}
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(argc > 0 ? a[1] : a[2])))]] // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(argc + k)))]]
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(argc), num_teams (argc+1)))]] // expected-error {{directive '#pragma omp teams distribute parallel for simd' cannot contain more than one 'num_teams' clause}}
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(S1)))]] // expected-error {{'S1' does not refer to a value}}
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(-2)))]] // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(-10u)))]]
  for (int i=0; i<100; i++) foo();
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams(3.14)))]] // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'double'}}
  for (int i=0; i<100; i++) foo();

  return 0;
}

int main(int argc, char **argv) {
  int k;
  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams))]] // expected-error {{expected '(' after 'num_teams'}}
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams ()))]] // expected-error {{expected expression}}
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams (argc > 0 ? argv[1] : argv[2])))]] // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams (argc + k)))]]
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams (argc), num_teams (argc+1)))]] // expected-error {{directive '#pragma omp teams distribute parallel for simd' cannot contain more than one 'num_teams' clause}}
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams (S1)))]] // expected-error {{'S1' does not refer to a value}}
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams (-2)))]] // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams (-10u)))]]
  for (int i=0; i<100; i++) foo();

  [[omp::sequence(directive(target), directive(teams distribute parallel for simd num_teams (3.14)))]] // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  for (int i=0; i<100; i++) foo();

  return tmain<int, 10>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 10>' requested here}}
}
