// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s

void foo();

int main(int argc, char **argv) {
#pragma omp task default                          // expected-error {{expected '(' after 'default'}}
#pragma omp task default(                         // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp task default()                        // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
#pragma omp task default(none                     // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp task default(shared), default(shared) // expected-error {{directive '#pragma omp task' cannot contain more than one 'default' clause}}
#pragma omp task default(x)                       // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  foo();

#pragma omp task default(none)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

#pragma omp task default(none)
#pragma omp task default(shared)
  ++argc;
  return 0;
}
