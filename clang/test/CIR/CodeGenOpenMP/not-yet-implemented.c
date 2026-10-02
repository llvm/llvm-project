// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fclangir %s -verify -emit-cir -o -

void do_things() {
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPCriticalDirective}}
#pragma omp critical
  {}

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPSingleDirective}}
#pragma omp single
  {}

  int i;
  // A directive that reports a not-yet-implemented clause emits no op at all,
  // rather than one that silently ignores the clause.
  int a, b;
  // expected-error@+2{{ClangIR code gen Not Yet Implemented: OpenMP PARALLEL 'shared' clause}}
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP PARALLEL 'firstprivate' clause}}
#pragma omp parallel shared(a) firstprivate(b)
  {}

  // A clause routed through construct decomposition but not yet emittable must
  // still be diagnosed by the directive emitter's NYI handling.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP TARGET 'private' clause}}
#pragma omp target private(i)
  {}

  // Decomposition can also synthesize a clause that the user did not write and
  // that has no Clang AST node to drive its emitter.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP synthesized 'firstprivate' clause from construct decomposition}}
#pragma omp target
  { i = 1; }

  // The teams construct supports no clauses yet, so every `teams` clause is
  // reported as not-yet-implemented.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP TEAMS 'num_teams' clause}}
#pragma omp teams num_teams(4)
  {}

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP TEAMS 'thread_limit' clause}}
#pragma omp teams thread_limit(8)
  {}
}
