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
    during(i);
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: {{.*}} = cir.load align(4) %{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: cir.call @during(%{{.*}}) : (!s32i) -> ()
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
   a = a + 1;
   b = b + 1;
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: cir.load align(4) %{{.*}}
  // CHECK-NEXT: cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: cir.binop(add, %{{.*}}, %{{.*}}) nsw : !s32i
  // CHECK-NEXT: cir.store align(4) %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.load align(4) %{{.*}}
  // CHECK-NEXT: cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: cir.binop(add, %{{.*}}, %{{.*}}) nsw : !s32i
  // CHECK-NEXT: cir.store align(4) %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}
void proc_bind_parallel() {
  // CHECK: cir.func{{.*}}@proc_bind_parallel
#pragma omp parallel proc_bind(master)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(master) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(close)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(close) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(spread)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(spread) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(primary)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(primary) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(default)
  {}
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}
