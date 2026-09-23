// AST dump + ast-print round-trip for omp_fill at every position in counts().
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump %s | FileCheck %s --check-prefix=DUMP
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

extern "C" void body(...);

// --- omp_fill at last position: counts(2, omp_fill) ---
// PRINT-LABEL: void fill_last(
// DUMP-LABEL:  FunctionDecl {{.*}} fill_last
void fill_last(int n) {
  // PRINT:     #pragma omp split counts(2, omp_fill)
  // DUMP:      OMPSplitDirective
  // DUMP:        OMPCountsClause
  #pragma omp split counts(2, omp_fill)
  // PRINT: for (int i = 0; i < n; ++i)
  // DUMP:      ForStmt
  for (int i = 0; i < n; ++i)
    body(i);
}

// --- omp_fill at first position: counts(omp_fill, 3) ---
// PRINT-LABEL: void fill_first(
// DUMP-LABEL:  FunctionDecl {{.*}} fill_first
void fill_first(int n) {
  // PRINT:     #pragma omp split counts(omp_fill, 3)
  // DUMP:      OMPSplitDirective
  // DUMP:        OMPCountsClause
  #pragma omp split counts(omp_fill, 3)
  // PRINT: for (int i = 0; i < n; ++i)
  // DUMP:      ForStmt
  for (int i = 0; i < n; ++i)
    body(i);
}

// --- omp_fill at middle position: counts(1, omp_fill, 1) ---
// PRINT-LABEL: void fill_mid(
// DUMP-LABEL:  FunctionDecl {{.*}} fill_mid
void fill_mid(int n) {
  // PRINT:     #pragma omp split counts(1, omp_fill, 1)
  // DUMP:      OMPSplitDirective
  // DUMP:        OMPCountsClause
  #pragma omp split counts(1, omp_fill, 1)
  // PRINT: for (int i = 0; i < n; ++i)
  // DUMP:      ForStmt
  for (int i = 0; i < n; ++i)
    body(i);
}

// --- omp_fill as sole item: counts(omp_fill) ---
// PRINT-LABEL: void fill_only(
// DUMP-LABEL:  FunctionDecl {{.*}} fill_only
void fill_only(int n) {
  // PRINT:     #pragma omp split counts(omp_fill)
  // DUMP:      OMPSplitDirective
  // DUMP:        OMPCountsClause
  #pragma omp split counts(omp_fill)
  // PRINT: for (int i = 0; i < n; ++i)
  // DUMP:      ForStmt
  for (int i = 0; i < n; ++i)
    body(i);
}

#endif
