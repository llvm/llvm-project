// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fclangir %s -verify -emit-cir -o -

void clash(void);

void do_things() {
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

  // A not-yet-implemented construct nested one level inside another
  // construct's body (rather than directly at the top level of a function)
  // must be diagnosed cleanly.
#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPCriticalDirective}}
#pragma omp critical
    {}
  }
}

void use_clash() {
  clash();
  // A critical name shares the symbol table with functions.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP critical name that clashes with another symbol}}
#pragma omp critical(clash)
  {}
}

void conflicting_hint() {
  // Sema accepts uncontended (1) together with contended (2), but the omp
  // dialect rejects it.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP critical 'hint' combining conflicting sync hints}}
#pragma omp critical(conflict) hint(3)
  {}
}

void loop_continue(int *a) {
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: continue in an OpenMP loop body}}
    if (i == 3) continue;
    a[i] = i;
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
