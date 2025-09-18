// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
  int z;
  #pragma omp target parallel nowait( //  expected-error {{expected expression}} //  expected-error {{expected ')'}} // expected-note {{to match this '('}}
    foo();
  #pragma omp target parallel nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel nowait device (-10u)
  foo();
  #pragma omp target parallel nowait (3.14) device (-10u)
  foo();
  #pragma omp target parallel nowait (argc>> z)
  foo();
  #pragma omp target parallel nowait (argv[1] = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel nowait (argc > 0 ? argv[1] : argv[2])
  foo();
  #pragma omp target parallel nowait (S1) // expected-error {{use of undeclared identifier 'S1'}}
  foo();

  return 0;
}
