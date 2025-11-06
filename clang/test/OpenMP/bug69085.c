// RUN: %clang_cc1 -verify -O0 -fopenmp-simd %s

int k[-1]; // expected-error {{'k' declared as an array with a negative size}}

void foo() {
    #pragma omp task depend(inout: k [:])
    {
        k[0] = 1;
    }
}
