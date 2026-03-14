// RUN: %clang_cc1 -verify -Wsource-uses-openmp -o - %s

// RUN: %clang_cc1 -verify -Wsource-uses-openmp -o - %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

int a;
#pragma omp threadprivate(a,b) // expected-warning {{unexpected '#pragma omp ...' in program}}
#pragma omp parallel
