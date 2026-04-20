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

void nowait() {
  int A=1;

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPNowaitClause
  // PRINT: #pragma omp target nowait
  #pragma omp target nowait
  {
  }

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPNowaitClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp target nowait(false)
  #pragma omp target nowait(false)
  {
  }

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPNowaitClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp target nowait(true)
  #pragma omp target nowait(true)
  {
  }

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPNowaitClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp target nowait(A > 5)
  #pragma omp target nowait(A>5)
  {
  }

}
#endif
