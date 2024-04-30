// RUN: %clang_cc1 -verify -fopenmp -ast-print -std=c++20 %s -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++20 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print -std=c++20 %s -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++20 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK:      void do_loop(const auto &v) {
// CHECK-NEXT: #pragma omp parallel for
// CHECK-NEXT:    for (const auto &i : v)
// CHECK-NEXT:      ;
// CHECK-NEXT: }

void do_loop(const auto &v) {
#pragma omp parallel for
  for (const auto &i : v)
    ;
}
#endif
