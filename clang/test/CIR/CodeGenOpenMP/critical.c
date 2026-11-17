// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// C++-specific cases (lambdas, references, templates) are in critical.cpp.

void work(void);
void work2(void);

void unnamed(void) {
#pragma omp critical
  work();
}

void named(void) {
#pragma omp critical(mylock)
  work();
}

void named_hint(void) {
#pragma omp critical(fast) hint(2)
  work();
}

// The same name in another function reuses the one declaration.
void named_again(void) {
#pragma omp critical(mylock)
  work2();
}

void no_braces(int *p) {
#pragma omp critical(mylock)
  *p += 1;
}

void local_var(int *p) {
#pragma omp critical
  {
    int tmp = *p;
    *p = tmp + 1;
  }
}

void in_parallel(int *p) {
#pragma omp parallel
  {
#pragma omp critical(mylock)
    { *p += 1; }
  }
}

void nested(void) {
#pragma omp critical(outer)
  {
    work();
#pragma omp critical(inner)
    work2();
  }
}

// Combinations with other directives.

void parallel_no_braces(void) {
#pragma omp parallel
#pragma omp critical
  work();
}

void parallel_clauses(int n, int cond) {
#pragma omp parallel num_threads(n) if(cond)
  {
#pragma omp critical(mylock)
    work();
  }
}

void barrier_between(void) {
#pragma omp parallel
  {
#pragma omp critical
    work();
#pragma omp barrier
#pragma omp critical(mylock)
    work2();
  }
}

void parallel_in_critical(void) {
#pragma omp critical(outer)
  {
#pragma omp parallel
    work();
  }
}

void target_parallel_critical(void) {
#pragma omp target parallel
  {
#pragma omp critical
    work();
  }
}

// Each name is declared once, at the start of the module.
// CIR-DAG: omp.critical.declare @mylock
// CIR-DAG: omp.critical.declare @fast hint(contended)
// CIR-DAG: omp.critical.declare @outer
// CIR-DAG: omp.critical.declare @inner

// CIR-LABEL: cir.func {{.*}}@unnamed()
// CIR:         omp.critical {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.terminator
// CIR:         }
// CIR:         cir.return

// CIR-LABEL: cir.func {{.*}}@named()
// CIR:         omp.critical(@mylock) {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@named_hint()
// CIR:         omp.critical(@fast) {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@named_again()
// CIR:         omp.critical(@mylock) {
// CIR:           cir.call @work2() : () -> ()
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@no_braces(
// CIR:         omp.critical(@mylock) {
// CIR:           cir.load
// CIR:           cir.add
// CIR:           cir.store
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@local_var(
// CIR:         omp.critical {
// CIR:           cir.alloca "tmp"
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@in_parallel(
// CIR:         omp.parallel {
// CIR:           omp.critical(@mylock) {
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@nested()
// CIR:         omp.critical(@outer) {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.critical(@inner) {
// CIR:             cir.call @work2() : () -> ()
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@parallel_no_braces()
// CIR:         omp.parallel {
// CIR:           omp.critical {
// CIR:             cir.call @work() : () -> ()
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@parallel_clauses(
// CIR:         omp.parallel if(%{{.*}}) num_threads(%{{.*}} : i32) {
// CIR:           omp.critical(@mylock) {
// CIR:             cir.call @work() : () -> ()
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@barrier_between()
// CIR:         omp.parallel {
// CIR:           omp.critical {
// CIR:             cir.call @work() : () -> ()
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.barrier
// CIR:           omp.critical(@mylock) {
// CIR:             cir.call @work2() : () -> ()
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@parallel_in_critical()
// CIR:         omp.critical(@outer) {
// CIR:           omp.parallel {
// CIR:             cir.call @work() : () -> ()
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@target_parallel_critical()
// CIR:         omp.target
// CIR:           omp.parallel {
// CIR:             omp.critical {
// CIR:               cir.call @work() : () -> ()
// CIR:               omp.terminator
// CIR:             }
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// The lock variables are shared by every critical section of the same name.
// Unlike classic codegen, a named critical section always goes through
// __kmpc_critical_with_hint, with a hint of 0 when none was given.
// LLVM-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[MYLOCK:.gomp_critical_user_mylock.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[FAST:.gomp_critical_user_fast.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[OUTER:.gomp_critical_user_outer.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[INNER:.gomp_critical_user_inner.var]] = {{.*}}global [8 x i32] zeroinitializer

// LLVM-LABEL: define {{.*}}void @unnamed()
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @named()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @named_hint()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[FAST]], i32 2)
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[FAST]])

// LLVM-LABEL: define {{.*}}void @named_again()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @work2()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @no_braces(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @local_var(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @in_parallel(
// LLVM:         call {{.*}}@__kmpc_fork_call(
// LLVM-LABEL: define {{.*}}void @in_parallel..omp_par(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @nested()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[OUTER]], i32 0)
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[INNER]], i32 0)
// LLVM:         call void @work2()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[INNER]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])

// LLVM-LABEL: define {{.*}}void @parallel_no_braces()
// LLVM:         call {{.*}}@__kmpc_fork_call(
// LLVM-LABEL: define {{.*}}void @parallel_no_braces..omp_par(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @parallel_clauses(
// LLVM:         call {{.*}}@__kmpc_fork_call_if(
// LLVM-LABEL: define {{.*}}void @parallel_clauses..omp_par(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @barrier_between()
// LLVM:         call {{.*}}@__kmpc_fork_call(
// LLVM-LABEL: define {{.*}}void @barrier_between..omp_par(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_barrier(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @work2()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// The parallel region runs while the calling thread holds the lock.
// LLVM-LABEL: define {{.*}}void @parallel_in_critical()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[OUTER]], i32 0)
// LLVM:         call {{.*}}@__kmpc_fork_call(
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])
// LLVM-LABEL: define {{.*}}void @parallel_in_critical..omp_par(
// LLVM:         call void @work()

// LLVM-LABEL: define {{.*}}void @target_parallel_critical()
// LLVM-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_target_parallel_critical_l{{[0-9]+}}(
// LLVM:         call {{.*}}@__kmpc_fork_call(
// LLVM-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_target_parallel_critical_l{{[0-9]+}}..omp_par(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[MYLOCK:.gomp_critical_user_mylock.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[FAST:.gomp_critical_user_fast.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[OUTER:.gomp_critical_user_outer.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[INNER:.gomp_critical_user_inner.var]] = {{.*}}global [8 x i32] zeroinitializer

// OGCG-LABEL: define {{.*}}void @unnamed()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @named()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @named_hint()
// OGCG:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[FAST]], i32 2)
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[FAST]])

// OGCG-LABEL: define {{.*}}void @named_again()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @work2()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @no_braces(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @local_var(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @in_parallel(
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @in_parallel.omp_outlined(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @nested()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[INNER]])
// OGCG:         call void @work2()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[INNER]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])

// OGCG-LABEL: define {{.*}}void @parallel_no_braces()
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @parallel_no_braces.omp_outlined(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @parallel_clauses(
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @parallel_clauses.omp_outlined(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @barrier_between()
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @barrier_between.omp_outlined(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_barrier(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @work2()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @parallel_in_critical()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])
// OGCG-LABEL: define {{.*}}void @parallel_in_critical.omp_outlined(
// OGCG:         call void @work()

// OGCG-LABEL: define {{.*}}void @target_parallel_critical()
// OGCG-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_target_parallel_critical_l{{[0-9]+}}(
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_target_parallel_critical_l{{[0-9]+}}.omp_outlined(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
