// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -ast-dump  %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix=PRINT

// Check same results after serialization round-trip
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -include-pch %t -ast-print    %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...);

// PRINT-LABEL: void foo1(
// DUMP-LABEL:  FunctionDecl {{.*}} foo1
void foo1() {
  // PRINT: #pragma omp interchange
  // DUMP:  OMPInterchangeDirective
  #pragma omp interchange
  // PRINT: for (int i = 7; i < 17; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT: for (int j = 7; j < 17; j += 3)
    // DUMP:  ForStmt
    for (int j = 7; j < 17; j += 3)
      // PRINT: body(i, j);
      // DUMP:  CallExpr
      body(i, j);
}


// PRINT-LABEL: void foo2(
// DUMP-LABEL:  FunctionDecl {{.*}} foo2
void foo2(int start1, int start2, int start3, int end1, int end2, int end3) {
  // PRINT:     #pragma omp interchange permutation(2, 3, 1)
  // DUMP:      OMPInterchangeDirective
  // DUMP-NEXT:   OMPPermutationClause
  // DUMP-NEXT:     ConstantExpr
  // DUMP-NEXT:       value: Int 2
  // DUMP-NEXT:       IntegerLiteral {{.*}} 2
  // DUMP-NEXT:     ConstantExpr
  // DUMP-NEXT:       value: Int 3
  // DUMP-NEXT:       IntegerLiteral {{.*}} 3
  // DUMP-NEXT:     ConstantExpr
  // DUMP-NEXT:       value: Int 1
  // DUMP-NEXT:       IntegerLiteral {{.*}} 1
 #pragma omp interchange permutation(2,3,1)
  // PRINT: for (int i = start1; i < end1; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = start1; i < end1; i += 1)
    // PRINT: for (int j = start2; j < end2; j += 1)
    // DUMP:  ForStmt
    for (int j = start2; j < end2; j += 1)
    // PRINT: for (int k = start3; k < end3; k += 1)
    // DUMP:  ForStmt
      for (int k = start3; k < end3; k += 1)
      // PRINT: body(i, j, k);
      // DUMP:  CallExpr
      body(i, j, k);
}


// PRINT-LABEL: void foo3(
// DUMP-LABEL:  FunctionDecl {{.*}} foo3
void foo3() {
  // PRINT: #pragma omp for collapse(3)
  // DUMP:      OMPForDirective
  // DUMP-NEXT:   OMPCollapseClause
  // DUMP-NEXT:     ConstantExpr
  // DUMP-NEXT:       value: Int 3
  // DUMP-NEXT:     IntegerLiteral {{.*}} 3
  // DUMP-NEXT:     CapturedStmt
  // DUMP-NEXT:       CapturedDecl
  #pragma omp for collapse(3)
  // PRINT: #pragma omp interchange
  // DUMP:  OMPInterchangeDirective
  #pragma omp interchange
  // PRINT: for (int i = 7; i < 17; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 1)
    // PRINT: for (int j = 7; j < 17; j += 1)
    // DUMP:  ForStmt
    for (int j = 7; j < 17; j += 1)
      // PRINT: for (int k = 7; k < 17; k += 1)
      // DUMP:  ForStmt
      for (int k = 7; k < 17; k += 1)
        // PRINT: body(i, j, k);
        // DUMP:  CallExpr
        body(i, j, k);
}


// PRINT-LABEL: void foo4(
// DUMP-LABEL:  FunctionDecl {{.*}} foo4
void foo4(int start, int end, int step) {
  // PRINT: #pragma omp for collapse(3)
  // DUMP:      OMPForDirective
  // DUMP-NEXT:   OMPCollapseClause
  // DUMP-NEXT:    ConstantExpr
  // DUMP-NEXT:      value: Int 3
  // DUMP-NEXT:    IntegerLiteral {{.*}} 3
  // DUMP-NEXT:  CapturedStmt
  // DUMP-NEXT:    CapturedDecl
  #pragma omp for collapse(3)
  // PRINT:     for (int i = 7; i < 17; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 1)
    // PRINT:     #pragma omp interchange permutation(1)
    // DUMP:      OMPInterchangeDirective
    // DUMP-NEXT:   OMPPermutationClause
    // DUMP-NEXT:     ConstantExpr
    // DUMP-NEXT:       value: Int 1
    // DUMP-NEXT:       IntegerLiteral {{.*}} 1
    #pragma omp interchange permutation(1)
    // PRINT:     for (int j = 7; j < 17; j += 1)
    // DUMP-NEXT: ForStmt
    for (int j = 7; j < 17; j += 1)
      // PRINT: for (int k = 7; k < 17; k += 1)
      // DUMP:  ForStmt
      for (int k = 7; k < 17; k += 1)
        // PRINT: body(i, j, k);
        // DUMP:  CallExpr
        body(i, j, k);
}


// PRINT-LABEL: void foo5(
// DUMP-LABEL: FunctionTemplateDecl {{.*}} foo5
template<typename T, T P>
void foo5(T start, T end) {
  // PRINT: #pragma omp for
  // DUMP:  OMPForDirective
  #pragma omp for
    // PRINT:     #pragma omp interchange permutation(P + 1, 2 - P)
    // DUMP:      OMPInterchangeDirective
    // DUMP-NEXT:   OMPPermutationClause
    // DUMP-NEXT:     BinaryOperator {{.*}} '+'
    // DUMP-NEXT:       DeclRefExpr {{.*}} 'P' 'T'
    // DUMP-NEXT:       IntegerLiteral {{.*}} 'int' 1
    // DUMP-NEXT:     BinaryOperator {{.*}} '-'
    // DUMP-NEXT:       IntegerLiteral {{.*}} 'int' 2
    // DUMP-NEXT:       DeclRefExpr {{.*}} 'P' 'T'
    #pragma omp interchange permutation(P + 1, 2 - P)
    // PRINT-NEXT: for (T i = start; i < end; i += 2)
    // DUMP-NEXT:  ForStmt
    for (T i = start; i < end; i += 2)
      // PRINT-NEXT: for (T j = start; j < end; j += 2)
      // DUMP:       ForStmt
      for (T j = start; j < end; j += 2)
        // PRINT-NEXT: body(i, j);
        // DUMP:       CallExpr
        body(i,j);
}

// Also test instantiating the template.
void tfoo5() {
  foo5<int,0>(0, 42);
  foo5<int,1>(0, 42);
}


// PRINT-LABEL: void foo6(
// DUMP-LABEL: FunctionTemplateDecl {{.*}} foo6
template<int Tile>
void foo6() {
    // PRINT:     #pragma omp interchange
    // DUMP:      OMPInterchangeDirective
    #pragma omp interchange
      // PRINT-NEXT: for (int i = 0; i < 11; i += 2)
      // DUMP-NEXT:  ForStmt
      for (int i = 0; i < 11; i += 2)
        // PRINT-NEXT: #pragma omp tile sizes(Tile)
        // DUMP:       OMPTileDirective
        #pragma omp tile sizes(Tile)
        // PRINT-NEXT: for (int j = 0; j < 13; j += 2)
        // DUMP:       ForStmt
        for (int j = 0; j < 13; j += 2)
          // PRINT-NEXT: body(i, j);
          // DUMP:       CallExpr
          body(i, j);
}

// Also test instantiating the template.
void tfoo6() {
  foo6<32>();
}


// PRINT-LABEL: void foo7(
// DUMP-LABEL: FunctionDecl {{.*}} foo7
void foo7() {
  double arr[128];
  // PRINT: #pragma omp interchange
  // DUMP:  OMPInterchangeDirective
  #pragma omp interchange
  // PRINT-NEXT: for (double c = 42; auto &&v : arr)
  // DUMP-NEXT:  CXXForRangeStmt
  for (double c = 42; auto &&v : arr)
    // PRINT-NEXT: for (int i = 0; i < 42; i += 2)
    // DUMP:       ForStmt
    for (int i = 0; i < 42; i += 2)
      // PRINT-NEXT: body(c, v, i);
      // DUMP:       CallExpr
      body(c, v, i);
}


// PRINT-LABEL: void foo8(
// DUMP-LABEL: FunctionDecl {{.*}} foo8
void foo8() {
  double arr[128];
  // PRINT: #pragma omp interchange
  // DUMP:  OMPInterchangeDirective
  #pragma omp interchange
  // PRINT-NEXT: for (int i = 0; i < 42; i += 2)
  // DUMP-NEXT:  ForStmt
  for (int i = 0; i < 42; i += 2)
    // PRINT-NEXT: for (double c = 42; auto &&v : arr)
    // DUMP:       CXXForRangeStmt
    for (double c = 42; auto &&v : arr)
      // PRINT-NEXT: body(i, c, v);
      // DUMP:       CallExpr
      body(i, c, v);
}

#endif

