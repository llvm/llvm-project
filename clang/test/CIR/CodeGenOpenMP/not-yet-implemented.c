// RUN: %clang_cc1 -fopenmp -fclangir %s -verify -emit-cir -o -

void do_things() {
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPCriticalDirective}}
#pragma omp critical
  {}

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPSingleDirective}}
#pragma omp single
  {}

  int i;
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPParallelDirective}}
#pragma omp parallel if(i)
  {}
}
