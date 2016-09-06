// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}} expected-note {{declared here}}

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  char ** argv;
  static T a;
// CHECK: static T a;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule // expected-error {{expected '(' after 'dist_schedule'}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule ( // expected-error {{expected 'static' in OpenMP clause 'dist_schedule'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule () // expected-error {{expected 'static' in OpenMP clause 'dist_schedule'}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (argc)) // expected-error {{expected 'static' in OpenMP clause 'dist_schedule'}} expected-warning {{extra tokens at the end of '#pragma omp distribute parallel for' are ignored}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, argc > 0 ? argv[1] : argv[2]) // expected-error2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static), dist_schedule (static, 1) // expected-error {{directive '#pragma omp distribute parallel for' cannot contain more than one 'dist_schedule' clause}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error3 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int i = 0; i < 10; ++i) foo();
  return T();
}

int main(int argc, char **argv) {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule // expected-error {{expected '(' after 'dist_schedule'}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule ( // expected-error {{expected 'static' in OpenMP clause 'dist_schedule'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule () // expected-error {{expected 'static' in OpenMP clause 'dist_schedule'}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (argc)) // expected-error {{expected 'static' in OpenMP clause 'dist_schedule'}} expected-warning {{extra tokens at the end of '#pragma omp distribute parallel for' are ignored}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static), dist_schedule (static, 1) // expected-error {{directive '#pragma omp distribute parallel for' cannot contain more than one 'dist_schedule' clause}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 0; i < 10; ++i) foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule (static, argv[1]=2) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0])); // expected-note {{in instantiation of function template specialization 'tmain<int, 5>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<char, 1>' requested here}}
}
