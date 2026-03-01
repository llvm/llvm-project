// RUN: %clang_cc1 -fopenmp -fclangir %s -verify -emit-cir -o -

void do_things() {
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPCriticalDirective}}
#pragma omp critical
  {}

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPSingleDirective}}
#pragma omp single
  {}

  int i;
  // TODO(OMP): We might consider overloading operator<< for OMPClauseKind in
  // the future if we want to improve this.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : if}}
#pragma omp parallel if(i)
  {}
}
