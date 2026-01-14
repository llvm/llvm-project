// RUN: not %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void before(int);
void during(int);
void after(int);

void emit_simple_parallel() {
  // CHECK: cir.func{{.*}}@emit_simple_parallel
  int i = 5;
  before(i);
  // CHECK: %[[I_LOAD:.*]] = cir.load{{.*}}
  // CHECK-NEXT: cir.call @before(%[[I_LOAD]])

#pragma omp parallel
  {}
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel
  {
    // TODO(OMP): We don't yet emit captured stmt, so the body of this is lost,x
    // thus we don't emit the 'during' call.
    during(i);
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

  after(i);
  // CHECK: %[[I_LOAD:.*]] = cir.load{{.*}}
  // CHECK-NEXT: cir.call @after(%[[I_LOAD]])
}

void parallel_with_operations() {
  // CHECK: cir.func{{.*}}@parallel_with_operations
  int a, b;
  // CHECK-NEXT: cir.alloca{{.*}}["a"]
  // CHECK-NEXT: cir.alloca{{.*}}["b"]
  // TODO(OMP): At the moment this results in 3 NYI diagnostics, 1 each for the
  // clauses + 1 for the CapturedStmt. When those are implemented, the check
  // lines will need updating.
#pragma omp parallel shared(a) firstprivate(b)
  {
   ++a;
   ++b;
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}
