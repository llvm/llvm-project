// Check no warnings/errors and that split is recognized
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST: OMPSplitDirective with associated for-loop
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump %s | FileCheck %s --check-prefix=DUMP

// Check unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

extern "C" void body(...);

// PRINT-LABEL: void foo(
// DUMP-LABEL:  FunctionDecl {{.*}} foo
void foo(int n) {
  // PRINT:     #pragma omp split counts(2, 3)
  // DUMP:      OMPSplitDirective
  // DUMP: OMPCountsClause
  #pragma omp split counts(2, 3)
  // PRINT: for (int i = 0; i < n; ++i)
  // DUMP:      ForStmt
  for (int i = 0; i < n; ++i)
    body(i);
}

#endif
