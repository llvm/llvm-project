// RUN: %clang_cc1 -verify -fopenmp -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -o - %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
#pragma omp target teams nowait( // expected-error {{expected expression}} // expected-error {{expected ')'}} //expected-note {{to match this '('}}
  foo();
#pragma omp target teams nowait device (-10u)
  foo();
#pragma omp target teams nowait (3.14) device (-10u) // expected-error {{arguments of OpenMP clause 'nowait' with bitwise operators cannot be of floating type}}
  foo();

  return 0;
}
