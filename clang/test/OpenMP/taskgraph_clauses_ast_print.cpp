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

void taskgraph_clauses() {
  int A = 1;

  // --- graph_id clause ---

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_idClause
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 0
  // PRINT: #pragma omp taskgraph graph_id(0)
  #pragma omp taskgraph graph_id(0)
  {}

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_idClause
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 42
  // PRINT: #pragma omp taskgraph graph_id(42)
  #pragma omp taskgraph graph_id(42)
  {}

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_idClause
  // DUMP: BinaryOperator {{.*}} 'int' '+'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 10
  // PRINT: #pragma omp taskgraph graph_id(A + 10)
  #pragma omp taskgraph graph_id(A + 10)
  {}

  // --- graph_reset clause ---

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_resetClause
  // DUMP-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' false
  // PRINT: #pragma omp taskgraph graph_reset(false)
  #pragma omp taskgraph graph_reset(false)
  {}

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_resetClause
  // DUMP-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true
  // PRINT: #pragma omp taskgraph graph_reset(true)
  #pragma omp taskgraph graph_reset(true)
  {}

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_resetClause
  // DUMP-NEXT: BinaryOperator {{.*}} 'bool' '>'
  // DUMP-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // DUMP-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'A' 'int'
  // DUMP-NEXT: IntegerLiteral {{.*}} 'int' 5
  // PRINT: #pragma omp taskgraph graph_reset(A > 5)
  #pragma omp taskgraph graph_reset(A > 5)
  {}

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_resetClause
  // PRINT: #pragma omp taskgraph graph_reset
  #pragma omp taskgraph graph_reset
  {}

  // --- Combinations using both clauses ---

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_idClause
  // DUMP: OMPGraph_resetClause
  // PRINT: #pragma omp taskgraph graph_id(1) graph_reset(true)
  #pragma omp taskgraph graph_id(1) graph_reset(true)
  {}

  // DUMP: OMPTaskgraphDirective
  // DUMP: OMPGraph_resetClause
  // DUMP: OMPGraph_idClause
  // PRINT: #pragma omp taskgraph graph_reset(false) graph_id(2)
  #pragma omp taskgraph graph_reset(false) graph_id(2)
  {}
}
#endif
