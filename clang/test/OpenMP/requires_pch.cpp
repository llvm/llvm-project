// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=x86_64 \
// RUN:   -triple x86_64 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=x86_64 \
// RUN:   -triple x86_64 -std=c++11 -include-pch %t -fsyntax-only %s

// expected-no-diagnostics

// A 'requires' directive read from an AST file must keep its effect on the
// translation unit including it.

#ifndef HEADER
#define HEADER
#pragma omp requires reverse_offload
void foo();
#else
void bar(int argc) {
#pragma omp target device(ancestor : argc)
  foo();
}
#endif
