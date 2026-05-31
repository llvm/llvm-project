// RUN: %clang_cc1 -fopenmp -fsyntax-only %s

void foo() {
#pragma omp taskloop transparent // expected-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}}
    for(int i = 0; i < 2; i++);
}
