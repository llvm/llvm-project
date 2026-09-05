// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -ast-dump  %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -ast-print %s | FileCheck %s --check-prefix=PRINT

// Check same results after serialization round-trip
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -include-pch %t -ast-print    %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...);

// The depth clause round-trips through -ast-print and PCH serialization.
// PRINT-LABEL: void foo_depth3(
// DUMP-LABEL:  FunctionDecl {{.*}} foo_depth3
void foo_depth3(int n, int m, int p) {
  // PRINT:     #pragma omp flatten depth(3)
  // DUMP:      OMPFlattenDirective
  // DUMP-NEXT: OMPDepthClause
  #pragma omp flatten depth(3)
  // PRINT: for (int i = 0; i < n; ++i)
  // DUMP:  ForStmt
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < p; ++k)
        // PRINT: body(i, j, k);
        // DUMP:  CallExpr
        body(i, j, k);
}

// depth(2) is accepted and printed like any other depth argument.
// PRINT-LABEL: void foo_depth2(
// DUMP-LABEL:  FunctionDecl {{.*}} foo_depth2
void foo_depth2(int n, int m) {
  // PRINT:     #pragma omp flatten depth(2)
  // DUMP:      OMPFlattenDirective
  // DUMP-NEXT: OMPDepthClause
  #pragma omp flatten depth(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      body(i, j);
}

// The depth clause is instantiated together with its enclosing template.
// PRINT-LABEL: template <typename T> void foo_tmpl()
// DUMP-LABEL:  FunctionTemplateDecl {{.*}} foo_tmpl
template <typename T>
void foo_tmpl() {
  // PRINT:     #pragma omp flatten depth(3)
  // DUMP:      OMPFlattenDirective
  // DUMP-NEXT: OMPDepthClause
  #pragma omp flatten depth(3)
  for (T i = 0; i < 8; ++i)
    for (T j = 0; j < 8; ++j)
      for (T k = 0; k < 8; ++k)
        body(i, j, k);
}

// PRINT-LABEL: template<> void foo_tmpl<int>()
// DUMP-LABEL:  FunctionDecl {{.*}} foo_tmpl 'void ()' implicit_instantiation
// DUMP:        OMPFlattenDirective
// DUMP-NEXT:   OMPDepthClause
void inst() {
  foo_tmpl<int>();
}

// depth as a non-type template parameter.
// PRINT-LABEL: template <int D> void foo_depth_d()
// DUMP-LABEL:  FunctionTemplateDecl {{.*}} foo_depth_d
template <int D>
void foo_depth_d() {
  // PRINT:     #pragma omp flatten depth(D)
  // DUMP:      OMPFlattenDirective
  // DUMP-NEXT: OMPDepthClause
  #pragma omp flatten depth(D)
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k)
        body(i, j, k);
}

// PRINT-LABEL: template<> void foo_depth_d<3>()
// DUMP-LABEL:  FunctionDecl {{.*}} foo_depth_d 'void ()' implicit_instantiation
// DUMP:        OMPFlattenDirective
// DUMP-NEXT:   OMPDepthClause
void inst_d() {
  foo_depth_d<3>();
}

#endif
