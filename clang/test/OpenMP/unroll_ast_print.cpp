// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -ast-dump  %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -ast-print %s | FileCheck %s --check-prefix=PRINT --match-full-lines

// Check same results after serialization round-trip
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -include-pch %t -ast-print    %s | FileCheck %s --check-prefix=PRINT --match-full-lines

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
void body(...);


// PRINT-LABEL: void func_unroll() {
// DUMP-LABEL:  FunctionDecl {{.*}} func_unroll
void func_unroll() {
  // PRINT:  #pragma omp unroll
  // DUMP:   OMPUnrollDirective
  #pragma omp unroll
  // PRINT-NEXT: for (int i = 7; i < 17; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT-NEXT: body(i);
    // DUMP: CallExpr
    body(i);
}


// PRINT-LABEL: void func_unroll_full() {
// DUMP-LABEL:  FunctionDecl {{.*}} func_unroll_full 
void func_unroll_full() {
  // PRINT:     #pragma omp unroll full
  // DUMP:      OMPUnrollDirective
  // DUMP-NEXT:   OMPFullClause
  #pragma omp unroll full
  // PRINT-NEXT: for (int i = 7; i < 17; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT-NEXT: body(i);
    // DUMP: CallExpr
    body(i);
}


// PRINT-LABEL: void func_unroll_partial() {
// DUMP-LABEL:  FunctionDecl {{.*}} func_unroll_partial 
void func_unroll_partial() {
  // PRINT:     #pragma omp unroll partial
  // DUMP:      OMPUnrollDirective
  // DUMP-NEXT:   OMPPartialClause
  // DUMP-NEXT:     <<<NULL>>>
  #pragma omp unroll partial
  // PRINT-NEXT: for (int i = 7; i < 17; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT: body(i);
    // DUMP: CallExpr
    body(i);
}


// PRINT-LABEL: void func_unroll_partial_factor() {
// DUMP-LABEL:  FunctionDecl {{.*}} func_unroll_partial_factor 
void func_unroll_partial_factor() {
  // PRINT:     #pragma omp unroll partial(4)
  // DUMP:      OMPUnrollDirective
  // DUMP-NEXT:   OMPPartialClause
  // DUMP-NEXT:     ConstantExpr
  // DUMP-NEXT:       value: Int 4
  // DUMP-NEXT:       IntegerLiteral {{.*}} 4
  #pragma omp unroll partial(4)
  // PRINT-NEXT: for (int i = 7; i < 17; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT-NEXT: body(i);
    // DUMP: CallExpr
    body(i);
}


// PRINT-LABEL: void func_unroll_partial_factor_for() {
// DUMP-LABEL:  FunctionDecl {{.*}} func_unroll_partial_factor_for 
void func_unroll_partial_factor_for() {
  // PRINT:     #pragma omp for
  // DUMP:      OMPForDirective
  #pragma omp for
  // PRINT:       #pragma omp unroll partial(2)
  // DUMP:        OMPUnrollDirective
  // DUMP-NEXT:     OMPPartialClause
  #pragma omp unroll partial(2)
  // PRINT-NEXT: for (int i = 7; i < 17; i += 3)
  // DUMP: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT-NEXT: body(i);
    // DUMP: CallExpr
    body(i);
}


// PRINT-LABEL: template <typename T, T Start, T End, T Step, int Factor> void unroll_templated() {
// DUMP-LABEL:  FunctionTemplateDecl {{.*}} unroll_templated
template<typename T, T Start, T End, T Step, int Factor>
void unroll_templated() {
  // PRINT: #pragma omp unroll partial(Factor)
  // DUMP:      OMPUnrollDirective
  // DUMP-NEXT: OMPPartialClause
  // DUMP-NEXT:   DeclRefExpr {{.*}} 'Factor' 'int'
  #pragma omp unroll partial(Factor)
    // PRINT-NEXT: for (T i = Start; i < End; i += Step)
    // DUMP-NEXT:  ForStmt
    for (T i = Start; i < End; i += Step)
      // PRINT-NEXT: body(i);
      // DUMP:  CallExpr
      body(i);
}
void unroll_template() {
  unroll_templated<int,0,1024,1,4>();
}


// PRINT-LABEL: template <int Factor> void unroll_templated_factor(int start, int stop, int step) {
// DUMP-LABEL:  FunctionTemplateDecl {{.*}} unroll_templated_factor
template <int Factor>
void unroll_templated_factor(int start, int stop, int step) {
  // PRINT: #pragma omp unroll partial(Factor)
  // DUMP:      OMPUnrollDirective
  // DUMP-NEXT: OMPPartialClause
  // DUMP-NEXT:   DeclRefExpr {{.*}} 'Factor' 'int'
  #pragma omp unroll partial(Factor)
    // PRINT-NEXT: for (int i = start; i < stop; i += step)
    // DUMP-NEXT:  ForStmt
    for (int i = start; i < stop; i += step)
      // PRINT-NEXT: body(i);
      // DUMP:  CallExpr
      body(i);
}
void unroll_template_factor() {
  unroll_templated_factor<4>(0, 42, 2);
}


#endif
