// RUN: %clang_cc1 -verify -fopenmp -frecovery-ast %s

void foo(int i) {
#pragma omp target update from(i) device(undef()) // expected-error {{use of undeclared identifier 'undef'}}
}
