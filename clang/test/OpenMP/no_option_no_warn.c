// RUN: %clang_cc1 -verify -Wno-source-uses-openmp -o - %s

// RUN: %clang_cc1 -verify -Wno-source-uses-openmp -o - %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

int a;
#pragma omp threadprivate(a, b)
#pragma omp parallel
