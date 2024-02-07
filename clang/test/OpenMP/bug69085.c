// RUN: %clang_cc1 -O0 -fopenmp-simd %s

int k[-46];

void foo() {
#pragma omp task depend(inout: k [0.5:])
}
