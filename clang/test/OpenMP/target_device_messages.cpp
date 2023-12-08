// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp51 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp51 -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

int main(int argc, char **argv) {
  int z;
  #pragma omp target device // expected-error {{expected '(' after 'device'}}
  foo();
  #pragma omp target device ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target device () // expected-error {{expected expression}}
  foo();
  #pragma omp target device (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target device (argc: // expected-error {{expected ')'}} expected-note {{to match this '('}} omp51-error {{expected expression}}
  foo();
  #pragma omp target device (z-argc)) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target device (device_num : argc > 0 ? argv[1] : argv[2]) // omp45-error {{use of undeclared identifier 'device_num'}} omp45-error {{expected ')'}} omp45-note {{to match this '('}} omp51-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target device (argc: argc + argc) // omp45-error {{expected ')'}} omp45-note {{to match this '('}} omp51-error {{expected 'ancestor' or 'device_num' in OpenMP clause 'device'}}
  foo();
  #pragma omp target device (argc), device (argc+1) // expected-error {{directive '#pragma omp target' cannot contain more than one 'device' clause}}
  foo();
  #pragma omp target device (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target device (-2) // expected-error {{argument to 'device' clause must be a non-negative integer value}}
  foo();
  #pragma omp target device (-10u)
  foo();
  #pragma omp target device (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  foo();
  #pragma omp target device (ancestor) // expected-error {{use of undeclared identifier 'ancestor'}}
  foo();

  return 0;
}
