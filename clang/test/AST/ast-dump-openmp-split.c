// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -fopenmp-version=60 -ast-dump %s | FileCheck %s
//
// OMPSplitDirective / OMPCountsClause;

void body(int);

void test(void) {
#pragma omp split counts(3, omp_fill)
  for (int i = 0; i < 10; ++i)
    body(i);
}

// CHECK: OMPSplitDirective
// CHECK: OMPCountsClause
// CHECK: IntegerLiteral{{.*}}3
// CHECK: <<<NULL>>>
// CHECK: ForStmt
// CHECK: <<<NULL>>>
// CHECK: CallExpr
