// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=52 -ast-dump %s | FileCheck %s --check-prefix=DUMP
// expected-no-diagnostics

// Check that the "update" clause is represented in AST by OMPUpdateClause
// when used on "atomic" construct.
int check_atomic(int x) {
  #pragma omp atomic update
  x = x + 1;

  return x;
}

// DUMP: FunctionDecl {{.*}} check_atomic 'int (int)' external-linkage
// DUMP: OMPAtomicDirective
// DUMP: OMPUpdateClause

using omp_depend_t = void *;

// Check that the "update" clause is represented in AST by
// OMPUpdateDependObjectsClause when used on "depobj" consturct.
void check_depobj(omp_depend_t x) {
  #pragma omp depobj(x) update(in)
  {}
}

// DUMP: FunctionDecl {{.*}} check_depobj 'void (omp_depend_t)' external-linkage
// DUMP: OMPDepobjDirective {{.*}} openmp_standalone_directive
// DUMP: OMPDepobjClause
// DUMP: OMPUpdateDependObjectsClause
