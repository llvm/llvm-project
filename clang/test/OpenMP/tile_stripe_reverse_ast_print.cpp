// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN: -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN: -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN: -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN: -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

extern "C" void body(...);

// PRINT-LABEL: void tile_then_reverse(
void tile_then_reverse() {
  // PRINT: #pragma omp tile sizes(2, 2)
  // PRINT: #pragma omp reverse
#pragma omp tile sizes(2, 2)
#pragma omp reverse
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 20; ++j)
      body(i, j);
}

// PRINT-LABEL: void stripe_then_reverse(
void stripe_then_reverse() {
  // PRINT: #pragma omp stripe sizes(2, 2)
  // PRINT: #pragma omp reverse
#pragma omp stripe sizes(2, 2)
#pragma omp reverse
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 20; ++j)
      body(i, j);
}

// PRINT-LABEL: void tile_over_inner_reverse(
void tile_over_inner_reverse() {
  // PRINT: #pragma omp tile sizes(2, 2)
#pragma omp tile sizes(2, 2)
#pragma omp reverse
  for (int j = 0; j < 20; ++j) {
    // PRINT: #pragma omp reverse
#pragma omp reverse
    for (int i = 0; i < 20; ++i)
      body(j, i);
  }
}

// PRINT-LABEL: void stripe_over_inner_reverse(
void stripe_over_inner_reverse() {
  // PRINT: #pragma omp stripe sizes(2, 2)
#pragma omp stripe sizes(2, 2)
#pragma omp reverse
  for (int j = 0; j < 20; ++j) {
    // PRINT: #pragma omp reverse
#pragma omp reverse
    for (int i = 0; i < 20; ++i)
      body(j, i);
  }
}

#endif
