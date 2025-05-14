// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

// Workshare is a Fortran-only directive.

void foo() {
#pragma omp workshare // expected-error {{expected an OpenMP directive}}
}

