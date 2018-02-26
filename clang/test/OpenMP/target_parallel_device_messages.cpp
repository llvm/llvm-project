// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -o - %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

int main(int argc, char **argv) {
  #pragma omp target parallel device // expected-error {{expected '(' after 'device'}}
  foo();
  #pragma omp target parallel device ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel device () // expected-error {{expected expression}}
  foo();
  #pragma omp target parallel device (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel device (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel device (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel device (argc + argc)
  foo();
  #pragma omp target parallel device (argc), device (argc+1) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'device' clause}}
  foo();
  #pragma omp target parallel device (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target parallel device (-2) // expected-error {{argument to 'device' clause must be a non-negative integer value}}
  foo();
  #pragma omp target parallel device (-10u)
  foo();
  #pragma omp target parallel device (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  foo();

  return 0;
}
