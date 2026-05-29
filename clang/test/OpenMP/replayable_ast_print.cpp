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

void replayable_clauses() {
  int A = 1;
  int B[10];

  // --- omp task ---

  // DUMP: OMPTaskDirective
  // DUMP-NEXT: OMPReplayableClause
  // PRINT: #pragma omp task replayable
  #pragma omp task replayable
  {}

  // DUMP: OMPTaskDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp task replayable(false)
  #pragma omp task replayable(false)
  {}

  // DUMP: OMPTaskDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp task replayable(true)
  #pragma omp task replayable(true)
  {}

  // DUMP: OMPTaskDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp task replayable(A > 5)
  #pragma omp task replayable(A > 5)
  {}

  // --- omp taskloop ---

  // DUMP: OMPTaskLoopDirective
  // DUMP-NEXT: OMPReplayableClause
  // PRINT: #pragma omp taskloop replayable
  #pragma omp taskloop replayable
  for (int i = 0; i < 10; ++i)
    {}

  // DUMP: OMPTaskLoopDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp taskloop replayable(false)
  #pragma omp taskloop replayable(false)
  for (int i = 0; i < 10; ++i)
    {}

  // DUMP: OMPTaskLoopDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp taskloop replayable(true)
  #pragma omp taskloop replayable(true)
  for (int i = 0; i < 10; ++i)
    {}

  // DUMP: OMPTaskLoopDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp taskloop replayable(A > 5)
  #pragma omp taskloop replayable(A > 5)
  for (int i = 0; i < 10; ++i)
    {}

  // --- omp taskwait ---

  // DUMP: OMPTaskwaitDirective
  // DUMP-NEXT: OMPReplayableClause
  // PRINT: #pragma omp taskwait replayable
  #pragma omp taskwait replayable

  // DUMP: OMPTaskwaitDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp taskwait replayable(false)
  #pragma omp taskwait replayable(false)

  // DUMP: OMPTaskwaitDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp taskwait replayable(true)
  #pragma omp taskwait replayable(true)

  // DUMP: OMPTaskwaitDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp taskwait replayable(A > 5)
  #pragma omp taskwait replayable(A > 5)

  // --- omp target ---

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPReplayableClause
  // PRINT: #pragma omp target replayable
  #pragma omp target replayable
  {}

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp target replayable(false)
  #pragma omp target replayable(false)
  {}

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp target replayable(true)
  #pragma omp target replayable(true)
  {}

  // DUMP: OMPTargetDirective
  // DUMP-NEXT: OMPReplayableClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp target replayable(A > 5)
  #pragma omp target replayable(A > 5)
  {}

  // --- omp target enter data ---

  // DUMP: OMPTargetEnterDataDirective
  // DUMP: OMPReplayableClause
  // PRINT: #pragma omp target enter data map(to: A) replayable
  #pragma omp target enter data map(to: A) replayable

  // DUMP: OMPTargetEnterDataDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp target enter data map(to: A) replayable(false)
  #pragma omp target enter data map(to: A) replayable(false)

  // DUMP: OMPTargetEnterDataDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp target enter data map(to: A) replayable(true)
  #pragma omp target enter data map(to: A) replayable(true)

  // DUMP: OMPTargetEnterDataDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp target enter data map(to: A) replayable(A > 5)
  #pragma omp target enter data map(to: A) replayable(A > 5)

  // --- omp target exit data ---

  // DUMP: OMPTargetExitDataDirective
  // DUMP: OMPReplayableClause
  // PRINT: #pragma omp target exit data map(from: A) replayable
  #pragma omp target exit data map(from: A) replayable

  // DUMP: OMPTargetExitDataDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp target exit data map(from: A) replayable(false)
  #pragma omp target exit data map(from: A) replayable(false)

  // DUMP: OMPTargetExitDataDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp target exit data map(from: A) replayable(true)
  #pragma omp target exit data map(from: A) replayable(true)

  // DUMP: OMPTargetExitDataDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp target exit data map(from: A) replayable(A > 5)
  #pragma omp target exit data map(from: A) replayable(A > 5)

  // --- omp target update ---

  // DUMP: OMPTargetUpdateDirective
  // DUMP: OMPReplayableClause
  // PRINT: #pragma omp target update to(A) replayable
  #pragma omp target update to(A) replayable

  // DUMP: OMPTargetUpdateDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp target update to(A) replayable(false)
  #pragma omp target update to(A) replayable(false)

  // DUMP: OMPTargetUpdateDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: XXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp target update to(A) replayable(true)
  #pragma omp target update to(A) replayable(true)

  // DUMP: OMPTargetUpdateDirective
  // DUMP: OMPReplayableClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp target update to(A) replayable(A > 5)
  #pragma omp target update to(A) replayable(A > 5)
}
#endif
