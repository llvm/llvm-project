// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fclangir %s -verify -emit-cir -o -

void do_things() {
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPCriticalDirective}}
#pragma omp critical
  {}

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPSingleDirective}}
#pragma omp single
  {}

  int i;
  // A leaf that reports a not-yet-implemented clause emits no op at all, rather
  // than one that silently ignores the clause.
  int a, b;
  // expected-error@+2{{ClangIR code gen Not Yet Implemented: OpenMP PARALLEL 'shared' clause}}
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP PARALLEL 'firstprivate' clause}}
#pragma omp parallel shared(a) firstprivate(b)
  {}

  // A clause routed through construct decomposition but not yet emittable must
  // still be diagnosed by the leaf emitter's NYI handling.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP TARGET 'private' clause}}
#pragma omp target private(i)
  {}

  // Decomposition can also synthesize a clause that the user did not write and
  // that has no Clang AST node to drive its emitter.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP synthesized 'firstprivate' clause from construct decomposition}}
#pragma omp target
  { i = 1; }
}

void loop_continue(int *a) {
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: continue in an OpenMP loop body}}
    if (i == 3) continue;
    a[i] = i;
  }

  // Diagnosed before the SPMD omp.target is created, so no target is left
  // behind without its omp.loop_nest.
  int s = 0;
#pragma omp target parallel for map(tofrom: s)
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: continue in an OpenMP loop body}}
    if (i == 3) continue;
  }

  // A continue that targets a loop nested in the body is supported.
#pragma omp for
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      if (j == 3) continue;
      a[j] = i;
    }
  }
}
