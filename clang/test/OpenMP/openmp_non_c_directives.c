// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

// Test the reaction to some Fortran-only directives.

void foo() {
#pragma omp allocators // expected-error {{expected an OpenMP directive}}
#pragma omp do // expected-error {{expected an OpenMP directive}}
#pragma omp end workshare // expected-error {{expected an OpenMP directive}}
#pragma omp parallel workshare // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
#pragma omp workshare // expected-error {{expected an OpenMP directive}}
}

