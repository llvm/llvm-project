// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump  %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix=PRINT

// Check same results after serialization round-trip
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -include-pch %t -ast-print    %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...);

// PRINT-LABEL: void foo1(
// DUMP-LABEL:  FunctionDecl {{.*}} foo1
void foo1() {
  // PRINT:     #pragma omp reverse
  // DUMP:      OMPReverseDirective
  #pragma omp reverse
  // PRINT: for (int i = 7; i < 17; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT: body(i);
    // DUMP:  CallExpr
      body(i);
}


// PRINT-LABEL: void foo2(
// DUMP-LABEL:  FunctionDecl {{.*}} foo2
void foo2(int start, int end, int step) {
  // PRINT:     #pragma omp reverse
  // DUMP:      OMPReverseDirective
  #pragma omp reverse
  // PRINT: for (int i = start; i < end; i += step)
  // DUMP-NEXT: ForStmt
  for (int i = start; i < end; i += step)
      // PRINT: body(i);
      // DUMP:  CallExpr
      body(i);
}


// PRINT-LABEL: void foo3(
// DUMP-LABEL:  FunctionDecl {{.*}} foo3
void foo3() {
  // PRINT: #pragma omp for
  // DUMP:  OMPForDirective
  // DUMP-NEXT:    CapturedStmt
  // DUMP-NEXT:      CapturedDecl
  #pragma omp for
  // PRINT:     #pragma omp reverse 
  // DUMP-NEXT: OMPReverseDirective
  #pragma omp reverse
  for (int i = 7; i < 17; i += 3)
    // PRINT: body(i);
    // DUMP:  CallExpr
    body(i);
}


// PRINT-LABEL: void foo4(
// DUMP-LABEL:  FunctionDecl {{.*}} foo4
void foo4() {
  // PRINT: #pragma omp for collapse(2)
  // DUMP: OMPForDirective
  // DUMP-NEXT: OMPCollapseClause
  // DUMP-NEXT:  ConstantExpr
  // DUMP-NEXT:    value: Int 2
  // DUMP-NEXT:  IntegerLiteral {{.*}} 2
  // DUMP-NEXT:    CapturedStmt
  // DUMP-NEXT:      CapturedDecl
  #pragma omp for collapse(2)
  // PRINT:     #pragma omp reverse
  // DUMP:      OMPReverseDirective
  #pragma omp reverse
  // PRINT: for (int i = 7; i < 17; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 1)
    // PRINT: for (int j = 7; j < 17; j += 1)
    // DUMP:  ForStmt
    for (int j = 7; j < 17; j += 1)
      // PRINT: body(i, j);
      // DUMP:  CallExpr
      body(i, j);
}


// PRINT-LABEL: void foo5(
// DUMP-LABEL:  FunctionDecl {{.*}} foo5
void foo5(int start, int end, int step) {
  // PRINT:     #pragma omp for collapse(2)
  // DUMP:      OMPForDirective
  // DUMP-NEXT:   OMPCollapseClause
  // DUMP-NEXT:    ConstantExpr
  // DUMP-NEXT:      value: Int 2
  // DUMP-NEXT:    IntegerLiteral {{.*}} 2
  // DUMP-NEXT:  CapturedStmt
  // DUMP-NEXT:    CapturedDecl
  #pragma omp for collapse(2)
  // PRINT:     for (int i = 7; i < 17; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 1)
    // PRINT: #pragma omp reverse
    // DUMP:  OMPReverseDirective
    #pragma omp reverse 
    // PRINT:     for (int j = 7; j < 17; j += 1)
    // DUMP-NEXT: ForStmt
    for (int j = 7; j < 17; j += 1)
      // PRINT: body(i, j);
      // DUMP:  CallExpr
      body(i, j);
}


// PRINT-LABEL: void foo6(
// DUMP-LABEL:  FunctionTemplateDecl {{.*}} foo6
template<typename T, T Step>
void foo6(T start, T end) {
  // PRINT: #pragma omp reverse
  // DUMP:  OMPReverseDirective
  #pragma omp reverse
    // PRINT-NEXT: for (T i = start; i < end; i += Step)
    // DUMP-NEXT:  ForStmt
    for (T i = start; i < end; i += Step)
      // PRINT-NEXT: body(i);
      // DUMP:       CallExpr
      body(i);
}

// Also test instantiating the template.
void tfoo6() {
  foo6<int,3>(0, 42);
}


// PRINT-LABEL: void foo7(
// DUMP-LABEL:  FunctionDecl {{.*}} foo7
void foo7() {
  double arr[128];
  // PRINT: #pragma omp reverse
  // DUMP:  OMPReverseDirective
  #pragma omp reverse
  // PRINT-NEXT: for (auto &&v : arr)
  // DUMP-NEXT:  CXXForRangeStmt
  for (auto &&v : arr)
    // PRINT-NEXT: body(v);
    // DUMP:       CallExpr
    body(v);
}

#endif

