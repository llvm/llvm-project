// RUN: %clang_cc1 -triple=x86_64 -verify -fopenmp -fopenmp-targets=x86_64 -x c++ -fexceptions -fcxx-exceptions %s
// RUN: %clang_cc1 -triple=x86_64 -verify -fopenmp-simd -fopenmp-targets=x86_64 -x c++ -fexceptions -fcxx-exceptions %s

void bar() {
#pragma omp target device(ancestor : 1) // expected-error {{Device clause with ancestor device-modifier used without specifying 'requires reverse_offload'}}
  ;
}
